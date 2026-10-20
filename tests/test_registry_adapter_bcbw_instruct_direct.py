"""The INSTRUCT-DIRECT tag on the full-suite capability readers (`bigcodebench_full` / `bfcl_full`; the declaration of 2026-09-25 19:13Z;
`_bcbw_common.INSTRUCT_DIRECT_RULE` / `INSTRUCT_ANCHOR_RULE`), on the SYNTHETIC full mirror of `_bcbw_full_fixtures` with `instruct=True`: the gemma4e4b family's
`s1` / `s1p` rows as before plus ONE `_it` store -- the instruct-checkpoint constitution_lora fit under `native/gemma4e4b-f22it/`, staged `itdoc__…` by the harness
roster's `it` selection, aliased `d1it__…` in the pins -- its gate record, its launch-set rows (the fitted row: selection `it`, block / transfer_kind
instruct_direct, seed_base null, class `arm`, anchor_tuple `s1/none`; the `it/none` row: class anchor, NO store) and its three-number rows. What it pins:

* THE GRAMMAR admits `<model key>_it` / `<model key>-f22it` / `bcbw-<model key>_it-<sha7>` / `gate_<model key>_it_…` beside the seed grammar, which reads as before;
  the tag reads in the family's s1 group; a family without the tag (nemotron) or a 200-item reader admits it for no selection;
* THE ROW files under `view_legacy gemma4e4b_it` (the spelling every instruct-direct row of the model carries in the registry), model gemma4e4b_it, transfer_kind
  DERIVED instruct_direct (fit_provenance.model_id = the served checkpoint; derive.transfer_kind; legacy_spellings.yaml agrees), seed_base null, trial 0, its object
  the harness roster's `it` staged id, its producer key `gemma4e4b_it/<store key>`;
* THE ANCHOR is the s1 none BY ALIAS: the contrast's anchor_run_id is the s1 none's run (anchor_kind sibling_none -- another standup's none, `anchor_kind_for`;
  cross_standup true, extra.anchor_alias); NO
  `_it` none run files; the alias is recorded on the status, never a not-run row;
* THE SEED-TAG RECORDS are byte for byte what they were without the `_it` rows;
* REFUSED BY NAME: a missing expected sha or pin key, a class the declaration does not carry, a store attesting another object, a `_it` none store, a row spelling a
  seed base or another block / transfer kind / anchor tuple, a base-checkpoint fit, fit_provenance.transfer true, an alias naming another object, a reader or a
  family that does not admit the tag.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import pytest

import _bcbw_allmodels_fixtures as Y
import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import bfcl_allmodels as BFA
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bfcl_seedrep as BFS
from sisteer.registry.adapters import bigcodebench_allmodels as BCA
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import bigcodebench_seedrep as BCS
from sisteer.registry.adapters import lineage_common as LC

MODULES = {"bigcodebench": BCF, "bfcl": BFF}
ADAPTERS = {"bigcodebench": BCF.BigcodebenchFullAdapter, "bfcl": BFF.BfclFullAdapter}
TAG, TAG_P, IT_TAG, IT_KEY, NONE, RELU = Z.TAG, Z.TAG_P, Z.IT_TAG, Z.IT_KEY, Z.NONE, Z.RELU
OTHER_SHA = hashlib.sha256(b"synthetic: another object entirely").hexdigest()


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _read(tmp_path, seeds, eval_id: str, **kw):
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, instruct=True, **kw)
    ad = Z.adapter(ADAPTERS[eval_id], fx, seeds)
    bt, facts = Z.read(ad)
    return fx, ad, bt, facts


def _it_runs(bt):
    return [r for r in bt["run"] if r["view_legacy"] == IT_TAG]


def test_the_grammar_admits_the_instruct_direct_tag_and_keeps_the_seed_grammar():
    assert IT_TAG == "gemma4e4b_it" and Z.IT_ROW_DIR == "gemma4e4b-f22it" and B.INSTRUCT_SELECTION == "it" and B.INSTRUCT_ANCHOR_SEED == 1
    for mk in ("gemma4e4b", "llama8b", "qwen35_4b"):
        assert B.MODEL_KEYS[mk].instruct_direct is True and B.tag_parts(f"{mk}_it") is None and B.model_key_of(f"{mk}_it") == mk and B.is_instruct_tag(f"{mk}_it")
        assert B.seed_tag_of(f"{mk}_it") == f"{mk}_s1" and B.tag_of_row_dir(f"{mk}-f22it") == f"{mk}_it" and B.row_dir_of(f"{mk}_it") == f"{mk}-f22it"
        assert B.launch_tag_of(f"{mk}_it", IT_KEY) == f"{mk}_it"
        assert B.SUFFIX_RE.match(f"bcbw-{mk}_it-0123abc").group("launch_tag") == f"{mk}_it" and B.SUFFIX_RE.match(f"bcbw-{mk}_it-0123abc-r2").group("relaunch") == "-r2"
        parts = B.gate_name_parts(f"gate_{mk}_it_{IT_KEY}_bcbw-{mk}_it-0123abc.json")
        assert parts == {"tag": f"{mk}_it", "arditi": None, "okey": IT_KEY, "suffix": f"bcbw-{mk}_it-0123abc"}
        assert B.group_tags(f"{mk}_s1", True) == {"s1": f"{mk}_s1", "s1p": f"{mk}_s1p", "it": f"{mk}_it"} and "it" not in B.group_tags(f"{mk}_s1", False)
    # nemotron runs no instruct-direct tag: the grammar reads the spelling (so a store is refused by name, never skipped) but no group admits it
    assert B.MODEL_KEYS["nemotron3nano4b"].instruct_direct is False and B.seed_tag_of("nemotron3nano4b_it") == "nemotron3nano4b_s1"
    assert "it" not in B.group_tags("nemotron3nano4b_s1", True)
    # the seed grammar, byte for byte: the qwen s2 / s3 groups never carry the tag (the anchor is the s1 none), the `main` alias and Arditi spellings read as before
    assert B.group_tags("qwen35_4b_s2", True) == {"s2": "qwen35_4b_s2", "s2p": "qwen35_4b_s2p"} and B.tag_parts("qwen35_4b_s2p") == (2, True)
    assert B.tag_of_row_dir("gemma4e4b-f22") == "gemma4e4b_s1" and B.seed_tag_of("gemma4e4b_main") == "gemma4e4b_s1" and B.tag_of_row_dir("qwen35_4b-f22") is None
    assert not B.SUFFIX_RE.match("bcbw-gemma4e4b_it_arditi-0123abc") and not B.TAG_RE.match("gemma4e4b_itx") and not B.ROW_DIR_RE.match("gemma4e4b-f22its1")
    assert B.gate_name_parts("gate_qwen35_4b_s1_arditi_arditi_allpos_bcbw-qwen35_4b_s1_arditi-0104b94.json")["arditi"] == "_arditi"
    with pytest.raises(KeyError):
        B.row_dir_of("gemma4e4b_main")


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_the_instruct_direct_row_files_under_its_own_tag_with_a_derived_transfer_kind_and_the_aliased_anchor(tmp_path, seeds, eval_id):
    fx, ad, bt, facts = _read(tmp_path, seeds, eval_id)
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and facts["not_run"] == [], facts["refusals"]
    assert facts["instruct_direct_tag"] == IT_TAG and facts["instruct_direct_rule"] == B.INSTRUCT_DIRECT_RULE and facts["instruct_anchor_rule"] == B.INSTRUCT_ANCHOR_RULE
    assert facts["selections"] == ["s1", "s1p"] and facts["tags"] == [TAG, TAG_P], "the seed selections' status spelling is unchanged"
    its = _it_runs(bt)
    assert len(its) == 1 and its[0]["campaign_cell_id_legacy"] == IT_KEY, "ONE instruct-direct run: the fitted key; NO `_it` none run"
    r = its[0]
    none_run = Y.run_at(bt, TAG, NONE)
    inst = seeds.instrument_row(MODULES[eval_id].INSTRUMENT_SEED, MODULES[eval_id].EVAL_ID, model_id="gemma4e4b_it")
    assert r["view_legacy"] == "gemma4e4b_it" and r["model_id"] == "gemma4e4b_it" and r["transfer_kind"] == "instruct_direct" and r["object_id"] == Z.IT_OBJ
    assert r["seed_base"] is None and r["trial_index"] == 0 and r["decode_seed_rule"] == "none" and r["instrument_id"] == inst["instrument_id"]
    assert r["item_set_id"] == MODULES[eval_id].ITEM_SET_ID and r["lane_source_legacy"] == MODULES[eval_id].SOURCE and r["producer_run_key_legacy"] == f"{IT_TAG}/{fx['it_key']}"
    assert r["run_id"] == MODULES[eval_id].run_id_for(IT_TAG, fx["it_key"], IT_KEY, inst["instrument_id"]) and r["run_id"] != Y.run_at(bt, TAG, RELU)["run_id"]
    fc = r["field_confidence_json"]["transfer_kind"]
    assert fc["confidence"] == "derived" and "derive.transfer_kind" in fc["source"] and "fit_model_id == run.model_id" in fc["source"] and "legacy_spellings.yaml" in fc["source"]
    x = r["extra"]
    idb = x["instruct_direct"]
    assert (idb["tag"], idb["selection"], idb["group"], idb["model_key"], idb["model_id"]) == (IT_TAG, "it", TAG, "gemma4e4b", "gemma4e4b_it")
    assert idb["fit_model_id_hf"] == idb["served_model_id_hf"] == Z.MODEL_HF and idb["fit_model_id"] == "gemma4e4b_it" and idb["fit_transfer"] is False
    assert idb["transfer_kind_derived"] == idb["transfer_kind_recorded"] == "instruct_direct" and idb["transfer_kind_derivation"] == "fit_model_id == run.model_id"
    assert (idb["pin_key"], idb["pin_key_alias"], idb["pin_key_alias_pinned"], idb["staged_id"], idb["harness_roster_selection"]) == (Z.IT_STAGED, Z.IT_ALIAS, True, Z.IT_STAGED, "it")
    assert idb["rule"] == B.INSTRUCT_DIRECT_RULE
    ls = x["launch_set"]
    assert (ls["class"], ls["block"], ls["transfer_kind"], ls["pin_key"], ls["object_sha256_expected"], ls["anchor_tuple"], ls["pin_key_alias"]) == \
        ("arm", "instruct_direct", "instruct_direct", Z.IT_STAGED, Z.IT_OBJ, "s1/none", Z.IT_ALIAS)
    assert x["seed_replicate_label"] is None and x["fit_seed"] is None and x["producer_view_legacy"] == "it" and x["producer_tag_legacy"] == IT_TAG and x["producer_row_dir_legacy"] == Z.IT_ROW_DIR
    assert x["object_route"]["route"] == "harness_roster_instruct_selection" and x["object_route"]["harness_roster_selection"] == "it" and x["object_route"]["staged_id"] == Z.IT_STAGED
    assert "instruct-direct tag gemma4e4b_it" in r["notes"] and "anchored by alias on gemma4e4b_s1/none" in r["notes"]
    # the anchor BY ALIAS (INSTRUCT_ANCHOR_RULE): the s1 none's run, cross-standup, the alias named on the run and the contrast
    alias = x["anchor_alias"]
    assert (alias["declared_anchor_tuple"], alias["alias_of"], alias["anchor_tag"], alias["anchor_object_key"], alias["own_none_store"]) == ("s1/none", f"{IT_TAG}/none", TAG, "none", False)
    blk = x["anchor"]
    assert blk["state"] == "contrasted" and blk["anchor_run_id"] == none_run["run_id"] and blk["anchor_view_legacy"] == TAG and blk["cross_standup"] is True
    assert blk["rule"] == B.INSTRUCT_ANCHOR_RULE and blk["anchor_alias"] == alias and blk["n_pairs"] == Z.N_FULL
    cs = [c for c in bt["run_contrast"] if c["run_id"] == r["run_id"]]
    assert len(cs) == 1
    c = cs[0]
    assert c["anchor_run_id"] == none_run["run_id"] and c["anchor_kind"] == "sibling_none" == B.anchor_kind_for({"cross_standup": True}) and c["n_pairs"] == Z.N_FULL
    assert {k["anchor_kind"] for k in bt["run_contrast"] if k["run_id"] != r["run_id"]} == {"same_standup_none"}, "the seed-tag rows pair with their OWN standup's none"
    assert c["extra"]["cross_standup"] is True and c["extra"]["anchor_view_legacy"] == TAG and c["extra"]["anchor_rule"] == B.INSTRUCT_ANCHOR_RULE and c["extra"]["anchor_alias"] == alias
    assert "BY ALIAS to gemma4e4b_s1/none" in c["extra"]["cross_standup_note"] and "own_none_state" not in c["extra"]
    p_s = Z.SPECS_IT[eval_id]["n_correct"] / Z.N_FULL
    p_a = Z.SPECS[eval_id][NONE]["n_correct"] / Z.N_FULL
    assert abs(c["delta"] - 100.0 * (p_s - p_a)) < 1e-9 and abs(c["extra"]["acc_steered_over_pairs"] - p_s) < 1e-9 and abs(c["extra"]["acc_anchor_over_pairs"] - p_a) < 1e-9
    # the alias on the status: the `it/none` launch row is recorded as the alias, never as a not-run row; the row facts name the derivation
    assert facts["anchor_aliases"] == [{"tuple_id": "it/none", "tag": IT_TAG, "object_key": NONE, "class": "anchor", "alias_of": f"{TAG}/none", "anchor_run_id": none_run["run_id"],
                                        "anchor_state": "contrasted", "declared_class_expected": "anchor", "declared_class_agrees": True,
                                        "launch_set_row": {"census_status": "run", "anchor_tuple": None, "block": "instruct_direct", "seed_base": None}, "rule": B.INSTRUCT_ANCHOR_RULE}]
    row = facts["rows"][f"{IT_TAG}|{IT_KEY}"]
    assert row["instruct_direct"] is True and row["transfer_kind"] == "instruct_direct" and row["anchor_alias_of"] == f"{TAG}/none" and row["paper_label"] == "constitution_lora"
    assert facts["contrasts"][f"{IT_TAG}|{IT_KEY}"]["anchor_alias"] is True and facts["contrasts"][f"{IT_TAG}|{IT_KEY}"]["cross_standup"] is True
    # the seed-tag rows beside it are what they were: base->instruct fits, no alias block, the none unchanged
    assert Y.run_at(bt, TAG, RELU)["transfer_kind"] == "base_to_instruct" and "anchor_alias" not in Y.run_at(bt, TAG, RELU)["extra"] and "instruct_direct" not in none_run["extra"]
    # the source state names the `_it` rows as members of the s1 group's launch set and three-number record
    states = ad.source_states(X.ctx(), ad.discover())
    ls_members = [m for s in states if s.json_path.startswith("$.evals.") for m in s.current_members]
    assert f"it/{IT_KEY}" in ls_members and "it/none" in ls_members
    tn_members = [m for s in states if s.json_path == "$.rows" for m in s.current_members]
    assert f"{IT_TAG}/{IT_KEY}" in tn_members


def test_the_seed_tag_records_are_byte_for_byte_what_they_were_without_the_instruct_direct_rows(tmp_path, seeds):
    def emission(sub: str, instruct: bool):
        d = tmp_path / sub
        d.mkdir()
        fx = Z.mirror(d, seeds, eval_id="bigcodebench", root=str(d / "mirror"), instruct=instruct)
        ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
        bt, facts = Z.read(ad)
        it_rids = {r["run_id"] for r in _it_runs(bt)}
        recs = {rt: [{k: v for k, v in r.items() if k != "record_id"} for r in rows if r.get("run_id") not in it_rids] for rt, rows in bt.items()}
        # the recount computation is one record per row, identical across rows (no run id): the `_it` row adds a copy, so the set is what is compared
        recs["computation"] = sorted({json.dumps(c, sort_keys=True, default=str) for c in recs.get("computation", [])})
        txt = json.dumps(recs, sort_keys=True, separators=(",", ":"), default=str).replace(fx["root"], "<ROOT>").replace(str(d), "<TMP>")
        return txt, facts, it_rids
    before, f0, it0 = emission("without", False)
    after, f1, it1 = emission("with", True)
    assert not it0 and len(it1) == 1 and f0["runs"] == 5 and f1["runs"] == 6
    assert before == after, "a seed-tag record moved when the instruct-direct rows were added"
    # the status differs by the instruct-direct keys alone (paths normalised: the two mirrors sit in different directories)
    only_new = {"instruct_direct_tag", "instruct_direct_rule", "instruct_anchor_rule", "anchor_aliases"}
    assert set(f1) - set(f0) == only_new

    # (the anchor currency's tallies count the `_it` row's own comparison beside the seed rows': one more own_current contrast, nothing else moves)
    tallies = {"anchor_states", "contrasts_filed", "contrasts_own"}
    assert (f1["anchor_states"], f1["contrasts_filed"], f1["contrasts_own"]) == ({"own_current": 4}, 4, 4) and (f0["anchor_states"], f0["contrasts_filed"], f0["contrasts_own"]) == ({"own_current": 3}, 3, 3)
    assert f0["contrasts_reanchored"] == f1["contrasts_reanchored"] == 0 and f0["contrasts_parked"] == f1["contrasts_parked"] == 0

    def status(f, sub):
        d = {k: v for k, v in f.items() if k not in only_new | tallies | {"runs", "rows", "contrasts"}}
        d["rows"] = {k: v for k, v in f["rows"].items() if not k.startswith(IT_TAG)}
        d["contrasts"] = {k: v for k, v in f["contrasts"].items() if not k.startswith(IT_TAG)}
        return json.dumps(d, sort_keys=True, default=str).replace(str(tmp_path / sub), "<TMP>")
    assert status(f0, "without") == status(f1, "with")


def _drop(key):
    def edit(lr):
        lr.pop(key, None)
    return edit


def _set(key, value):
    def edit(lr):
        lr[key] = value
    return edit


REFUSALS = {
    "missing_expected_sha": (dict(instruct_row_override=_set("object_sha256_expected", None)), "the launch set expects object None"),
    "missing_pin_key": (dict(instruct_row_override=_set("pin_key", None)), "pin key None is not the roster's staged id"),
    "class_prompted": (dict(instruct_row_override=_set("class", "prompted")), "class 'prompted', the key's grammar says 'arm'"),
    "class_composed": (dict(instruct_row_override=_set("class", "composed")), "class 'composed', the key's grammar says 'arm'"),
    "store_attests_another_object": (dict(instruct_attest_sha=OTHER_SHA), "is no prefix of the pinned object"),
    "seed_base_spelled": (dict(instruct_row_override=_set("seed_base", 1)), "null (instruct-direct)"),
    "block_not_instruct_direct": (dict(instruct_row_override=_set("block", "seed")), "declares (block 'seed', transfer_kind 'instruct_direct', anchor_tuple 's1/none')"),
    "transfer_kind_missing": (dict(instruct_row_override=_drop("transfer_kind")), "transfer_kind ''"),
    "anchor_tuple_other": (dict(instruct_row_override=_set("anchor_tuple", "s1p/none")), "anchor_tuple 's1p/none'"),
    "base_checkpoint_fit": (dict(instruct_fit_hf=Z.FIT_HF), f"fit_provenance.model_id '{Z.FIT_HF}' is not the served checkpoint"),
    "fit_transfer_true": (dict(instruct_fit_transfer=True), "fit_provenance.transfer is true"),
    "alias_names_another_object": (dict(instruct_alias_sha=OTHER_SHA), "an alias must spell the SAME object"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_declaration_the_row_does_not_carry_refuses_the_instruct_direct_store_by_name_and_the_seed_rows_file(tmp_path, seeds, case):
    kw, needle = REFUSALS[case]
    fx, ad, bt, facts = _read(tmp_path, seeds, "bigcodebench", **kw)
    why = Z.refused(facts, IT_TAG, IT_KEY)
    assert needle in why, (case, why)
    assert facts["runs"] == 5 and facts["n_refusals"] == 1 and not _it_runs(bt), (case, facts["refusals"])
    assert all(c["run_id"] != None for c in bt["run_contrast"]) and len(bt["run_contrast"]) == 3, "the seed rows' contrasts (the ReLU, the prompted, the composed) file as before"
    assert Y.run_at(bt, TAG, NONE)["run_id"] and facts["anchor_aliases"][0]["anchor_state"] == "contrasted"


def test_a_none_store_under_the_instruct_direct_tag_refuses_by_name_and_no_second_none_run_files(tmp_path, seeds):
    fx, ad, bt, facts = _read(tmp_path, seeds, "bigcodebench", instruct_none_store=True)
    why = Z.refused(facts, IT_TAG, NONE)
    assert "BY ALIAS to gemma4e4b_s1/none" in why and "no duplicate none run files" in why
    assert facts["runs"] == 6 and facts["n_refusals"] == 1 and [r["campaign_cell_id_legacy"] for r in _it_runs(bt)] == [IT_KEY]
    assert sum(1 for r in bt["run"] if r["campaign_cell_id_legacy"] == NONE) == 2, "the s1 none and the s1p none alone"
    # the fitted `_it` row still anchors on the s1 none by alias
    c = [c for c in bt["run_contrast"] if c["run_id"] == _it_runs(bt)[0]["run_id"]][0]
    assert c["anchor_run_id"] == Y.run_at(bt, TAG, NONE)["run_id"] and c["extra"]["cross_standup"] is True


def test_a_reader_or_a_family_that_does_not_admit_the_tag_refuses_the_store_by_name(tmp_path, seeds, monkeypatch):
    # (a) a reader whose spec does not admit the tag (the 200-item readers' setting), over the same mirror
    monkeypatch.setattr(BCF.BigcodebenchFullAdapter, "SPEC", dataclasses.replace(BCF.SPEC, instruct_direct=False))
    fx, ad, bt, facts = _read(tmp_path / "reader", seeds, "bigcodebench")
    why = Z.refused(facts, IT_TAG, IT_KEY)
    assert "admits for no selection of the group gemma4e4b_s1" in why and "s1 -> gemma4e4b_s1, s1p -> gemma4e4b_s1p" in why and "it -> " not in why
    assert facts["runs"] == 5 and not _it_runs(bt) and "instruct_direct_tag" not in facts and "anchor_aliases" not in facts and facts["not_run"] == []
    monkeypatch.undo()
    # (b) a family that runs no instruct-direct tag (nemotron's setting), under a reader that admits it
    monkeypatch.setitem(B.MODEL_KEYS, "gemma4e4b", dataclasses.replace(B.MODEL_KEYS["gemma4e4b"], instruct_direct=False))
    fx, ad, bt, facts = _read(tmp_path / "family", seeds, "bigcodebench")
    why = Z.refused(facts, IT_TAG, IT_KEY)
    assert "admits for no selection of the group gemma4e4b_s1" in why and "for a family that runs the tag" in why
    assert facts["runs"] == 5 and not _it_runs(bt)


def test_the_full_readers_are_at_four_and_the_200_item_readers_at_their_own_versions():
    # the instruct-direct admission took the full readers to 3 (2026-09-25 / 26); the anchor currency (2026-10-03, commit 7) moves EVERY reader of the family by one
    assert AB.emission_version_of(BCF.BigcodebenchFullAdapter) == "4" and AB.emission_version_of(BFF.BfclFullAdapter) == "4"
    assert BCF.SPEC.instruct_direct is True and BFF.SPEC.instruct_direct is True
    for mod, ver in ((BCA, "2"), (BFA, "2"), (BCS, "3"), (BFS, "3")):
        assert mod.SPEC.instruct_direct is False
    assert AB.emission_version_of(BCA.BigcodebenchAllmodelsAdapter) == "2" and AB.emission_version_of(BFA.BfclAllmodelsAdapter) == "2"
    assert AB.emission_version_of(BCS.BigcodebenchSeedrepAdapter) == "3" and AB.emission_version_of(BFS.BfclSeedrepAdapter) == "3"
    assert B.LAUNCH_CLASSES == ("anchor", "arm", "prompted", "composed", "arditi") and B.INSTRUCT_CLASS in B.LAUNCH_CLASSES
    assert B.INSTRUCT_TRANSFER_KIND == "instruct_direct" and B.INSTRUCT_BLOCK == "instruct_direct"


def test_an_instruct_direct_row_relaunched_at_another_tree_reads_its_r2_store_gated_by_its_own_record(tmp_path, seeds):
    """Four `_it` rows relaunched (`bcbw-<mk>_it-<sha7>-r2`) after a first attempt refused at standup with NO store; a FAIL
    record of that attempt may sit beside the live store's PASS record and must never shadow it."""
    fx, ad, bt, facts = _read(tmp_path, seeds, "bigcodebench", instruct_relaunched=True, instruct_parked_gate=True)
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and "parked" not in facts, facts["refusals"]
    r = _it_runs(bt)[0]
    assert r["extra"]["store_suffix_legacy"].endswith(f"-{Z.RELAUNCH_TAG}") and r["code_sha"] == Z.TREE2 and r["extra"]["served_tree_sha"] == Z.TREE2
    ls = r["extra"]["launch_set"]
    assert ls["sha"] == Z.TREE2 and ls["tree_bound"] == Z.TREE2 and ls["tree_source"] == "row sha" and ls["launched_as"]["suffix"] == r["extra"]["store_suffix_legacy"]
    assert ls["relaunched_from"]["suffix"] == f"bcbw-{IT_TAG}-{Z.TREE[:7]}" and ls["relaunch"]["tag"] == Z.RELAUNCH_TAG and ls["relaunch"]["sha"] == Z.TREE2 and ls["relaunch"]["rule"] == B.RELAUNCH_RULE
    assert ls["task_tag"] is None and "one_task_store_rule" not in ls
    assert r["extra"]["gate"]["record"].endswith(f"-{Z.RELAUNCH_TAG}.json") and r["extra"]["gate"]["verdict"] == "PASS" and r["extra"]["gate"]["cross_tree"] is True
    # the contrast still anchors on the s1 none by alias
    c = [c for c in bt["run_contrast"] if c["run_id"] == r["run_id"]][0]
    assert c["anchor_run_id"] == Y.run_at(bt, TAG, NONE)["run_id"] and c["extra"]["anchor_alias"]["anchor_tag"] == TAG
    # without relaunches.json the relaunched `_it` store refuses by name, as a seed row's would
    fx2, ad2, bt2, facts2 = _read(tmp_path / "norl", seeds, "bigcodebench", instruct_relaunched=True, no_relaunches=True)
    assert "names the relaunch gemma4e4b_it/" in Z.refused(facts2, IT_TAG, IT_KEY) and facts2["runs"] == 5


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_a_row_whose_two_tasks_live_in_two_one_task_stores_files_its_own_and_never_refuses_the_other(tmp_path, seeds, eval_id):
    """The qwen `_it` difference_lora row ran BFCL in one store (`bcbw-<mk>_it-4bb77ea`) and BigCodeBench in another
    (`bcbw-<mk>_it-acb7434-bcb`, launched_as with tag `bcb`): each source reads the store of its task and neither refuses the other for the missing task."""
    fx, ad, bt, facts = _read(tmp_path, seeds, eval_id, instruct_two_stores=True)
    assert facts["n_refusals"] == 0 and facts["runs"] == 6, facts["refusals"]
    r = _it_runs(bt)[0]
    tag_tail = Z.TWO_STORES_TASK_TAG[eval_id]
    sfx = r["extra"]["store_suffix_legacy"]
    if eval_id == "bigcodebench":
        assert sfx == f"bcbw-{IT_TAG}-{Z.TREE2[:7]}-bcb" and r["code_sha"] == Z.TREE2 and r["extra"]["launch_set"]["task_tag"] == "bcb"
        assert r["extra"]["launch_set"]["launched_as"] == {"wave": IT_TAG, "row_dir": Z.IT_ROW_DIR, "suffix": sfx, "sha": Z.TREE2, "tag": "bcb"}
        assert r["extra"]["launch_set"]["one_task_store_rule"] == B.ONE_TASK_STORE_RULE
        # the BFCL-only store under the bare suffix sits in this root too: parked as the other task's store, never refused
        assert [p["suffix"] for p in facts["parked"]] == [f"bcbw-{IT_TAG}-{Z.TREE[:7]}"] and facts["parked"][0]["rule"] == B.ONE_TASK_STORE_RULE and facts["n_parked"] == 1
        assert "holds no inspect/bigcodebench_full files" in facts["parked"][0]["why"]
    else:
        assert sfx == f"bcbw-{IT_TAG}-{Z.TREE[:7]}" and r["code_sha"] == Z.TREE and r["extra"]["launch_set"]["task_tag"] is None and r["extra"]["launch_set"]["launched_as"] is None
        assert "parked" not in facts, "the `-bcb` store is the other task's: neither discovered nor parked here"
        assert not any("-bcb" in os.path.relpath(i.uri, fx["root"]) for i in ad.discover()), "neither the `-bcb` store's files nor its gate record are this source's inputs"
    assert tag_tail == ("-bcb" if eval_id == "bigcodebench" else "")
    assert r["n_items_planned"] == Z.N_FULL and r["view_legacy"] == IT_TAG and r["transfer_kind"] == "instruct_direct"
    c = [c for c in bt["run_contrast"] if c["run_id"] == r["run_id"]][0]
    assert c["anchor_run_id"] == Y.run_at(bt, TAG, NONE)["run_id"] and c["n_pairs"] == Z.N_FULL
    assert B.suffix_task(f"bcbw-{IT_TAG}-{Z.TREE2[:7]}-bcb") == "bigcodebench" and B.suffix_task(f"bcbw-{IT_TAG}-{Z.TREE2[:7]}-r2-bfcl") == "bfcl" and B.suffix_task(sfx) in (None, eval_id)


#: The qwen `_it` difference_lora row's two one-task attempts -- the BigCodeBench-only `-bcb` store FAILED its gate, the BFCL-only
#: store PASSED but is superseded -- before ONE two-task `-r2` rerun; neither attempt is mirrored, both gate records ride under records/
PARKED_TWO = ({"tree": Z.TREE, "extra": "-bcb", "verdict": "FAIL", "shape": "bigcodebench"}, {"tree": Z.TREE3, "extra": "", "verdict": "PASS", "shape": "bfcl"})


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_a_two_task_rerun_after_two_parked_one_task_attempts_reads_the_r2_store_on_both_evals_and_lists_the_parked_records(tmp_path, seeds, eval_id):
    fx, ad, bt, facts = _read(tmp_path, seeds, eval_id, instruct_relaunched=True, instruct_parked=PARKED_TWO)
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and "parked" not in facts, facts["refusals"]
    r = _it_runs(bt)[0]
    live = f"bcbw-{IT_TAG}-{Z.TREE2[:7]}-{Z.RELAUNCH_TAG}"
    bcb_sfx, bare_sfx = f"bcbw-{IT_TAG}-{Z.TREE[:7]}-bcb", f"bcbw-{IT_TAG}-{Z.TREE3[:7]}"
    assert r["extra"]["store_suffix_legacy"] == live and r["code_sha"] == Z.TREE2 and r["extra"]["gate"]["record"].endswith(f"{live}.json") and r["extra"]["gate"]["verdict"] == "PASS"
    ls = r["extra"]["launch_set"]
    assert ls["launched_as"]["suffix"] == live and ls["relaunched_from"]["suffix"] == bcb_sfx and ls["relaunched_from"]["suffixes"] == [bcb_sfx, bare_sfx]
    assert ls["relaunch"]["tag"] == Z.RELAUNCH_TAG and ls["relaunch"]["sha"] == Z.TREE2 and ls["relaunch"]["parked_suffix"] == [bcb_sfx, bare_sfx]
    assert ls["task_tag"] is None
    # the parked attempts' records: neither the row's store nor a refusal, listed by name; the `-bcb` record is the BigCodeBench source's alone
    listed = facts["gate_records_without_store"]
    want = {bcb_sfx: "FAIL", bare_sfx: "PASS"} if eval_id == "bigcodebench" else {bare_sfx: "PASS"}
    assert {g["suffix"]: g["verdict"] for g in listed} == want and len(listed) == len(want)
    for g in listed:
        assert (g["tag"], g["object_key"], g["row_dir"], g["launched_as_suffix"], g["named_by_relaunched_from"], g["parked_by_relaunches"]) == (IT_TAG, IT_KEY, Z.IT_ROW_DIR, live, True, True)
        assert g["record"].endswith(f"{g['suffix']}.json") and g["rule"] == B.RELAUNCH_RULE
    # the seed rows carry none of it; the contrast anchors on the s1 none by alias
    assert all("gate_records_without_store" not in x for x in ()) and Y.run_at(bt, TAG, RELU)["extra"]["launch_set"].get("relaunch") is None
    c = [c for c in bt["run_contrast"] if c["run_id"] == r["run_id"]][0]
    assert c["anchor_run_id"] == Y.run_at(bt, TAG, NONE)["run_id"] and c["extra"]["cross_standup"] is True
    # the launch set's trees name the parked-only tree; the row binds to its own
    assert any(t["sha"] == Z.TREE3 for t in json.load(open(os.path.join(fx["root"], B.RECORDS_DIR, "launch_sets.json")))["trees"]) and ls["tree_bound"] == Z.TREE2


def test_a_mirrored_member_of_the_parked_list_is_parked_by_the_list_never_refused(tmp_path, seeds):
    """The BFCL-only attempt PASSED and is superseded; were its store mirrored beside the `-r2` rerun (it holds the BFCL task's files), the BFCL source parks it
    through relaunched_from.suffixes -- the relaunch rule's second-store refusal is for a store the row names NOWHERE."""
    parked = (PARKED_TWO[0], dict(PARKED_TWO[1], store=True))
    fx, ad, bt, facts = _read(tmp_path, seeds, "bfcl", instruct_relaunched=True, instruct_parked=parked)
    assert facts["n_refusals"] == 0 and facts["runs"] == 6, facts["refusals"]
    bare_sfx, live = f"bcbw-{IT_TAG}-{Z.TREE3[:7]}", f"bcbw-{IT_TAG}-{Z.TREE2[:7]}-{Z.RELAUNCH_TAG}"
    assert _it_runs(bt)[0]["extra"]["store_suffix_legacy"] == live
    assert [p["suffix"] for p in facts["parked"]] == [bare_sfx] and facts["parked"][0]["relaunched_to"] == live and facts["parked"][0]["parked_suffixes"] == [f"bcbw-{IT_TAG}-{Z.TREE[:7]}-bcb", bare_sfx]
    assert facts["parked"][0]["rule"] == B.RELAUNCH_RULE and facts["n_parked"] == 1
    assert "gate_records_without_store" not in facts, "the mirrored attempt's record has its store; the `-bcb` record is not this source's"
    # a second present store the row names NOWHERE still refuses by name (the relaunch rule stands): drop it from the lists
    def unlist(lr):
        lr["relaunched_from"] = {"suffix": f"bcbw-{IT_TAG}-{Z.TREE[:7]}-bcb", "why": "synthetic"}
    fx2, ad2, bt2, facts2 = _read(tmp_path / "unlisted", seeds, "bfcl", instruct_relaunched=True, instruct_parked=parked, instruct_row_override=unlist)
    why = Z.refused(facts2, IT_TAG, IT_KEY)
    assert ("the launch set says the row was launched at tree" in why or "the launch set says the row was launched as" in why) and facts2["runs"] == 6 and "parked" not in facts2


#: THE LIVE SHAPES after the first instruct-direct launches (2026-09-25 22:5xZ): the rerun rows carry `sha` = the FIRST attempt's tree while
#: `launched_as.sha` = the rerun's. qwen difference-LoRA: row.sha 4bb77ea (the BFCL-only PASS attempt, parked with the FAILED `-bcb` attempt; parked_suffix a LIST);
#: llama difference-LoRA: row.sha acb7434 (one INCOMPLETE two-task attempt; parked_suffix a bare STRING). Here TREE / TREE3 stand for the first attempts, TREE2 for b3836e6.
LIVE_QWEN = dict(instruct_relaunched=True, instruct_row_sha=Z.TREE,
                 instruct_parked=({"tree": Z.TREE, "extra": "", "verdict": "PASS", "shape": "bfcl"}, {"tree": Z.TREE3, "extra": "-bcb", "verdict": "FAIL", "shape": "bigcodebench"}))
LIVE_LLAMA = dict(instruct_relaunched=True, instruct_row_sha=Z.TREE3, instruct_parked=({"tree": Z.TREE3, "extra": "", "verdict": "FAIL", "shape": None},), instruct_parked_scalar=True)
LIVE_SHAPES = {"qwen_bigcodebench": ("bigcodebench", LIVE_QWEN, Z.TREE), "qwen_bfcl": ("bfcl", LIVE_QWEN, Z.TREE), "llama_bigcodebench": ("bigcodebench", LIVE_LLAMA, Z.TREE3)}


@pytest.mark.parametrize("shape", sorted(LIVE_SHAPES))
def test_a_row_whose_sha_is_not_the_launched_stores_tree_is_an_inconsistent_record_refused_by_name_and_the_corrected_row_files(tmp_path, seeds, shape):
    eval_id, kw, row_sha = LIVE_SHAPES[shape]
    live = f"bcbw-{IT_TAG}-{Z.TREE2[:7]}-{Z.RELAUNCH_TAG}"
    parked = [X.suffix_of(IT_TAG, IT_KEY, tree=p["tree"], suffix_extra=p.get("extra", "")) for p in kw["instruct_parked"]]
    # the live shape: row.sha = the first attempt's tree, launched_as.sha = the rerun's -- refused by name, nothing read for the row, never not-run
    fx, ad, bt, facts = _read(tmp_path / "live", seeds, eval_id, **kw)
    why = Z.refused(facts, IT_TAG, IT_KEY)
    assert f"the row's sha {row_sha[:12]} is not the launched store's tree {Z.TREE2[:12]}" in why and f"launched_as names '{live}' at {Z.TREE2[:12]}" in why
    assert "the record is inconsistent" in why and B.LAUNCHED_AS_TREE_RULE in why
    assert facts["runs"] == 5 and facts["n_refusals"] == 1 and facts["not_run"] == [] and not _it_runs(bt) and "parked" not in facts
    assert {g["suffix"] for g in facts["gate_records_without_store"]} == {p for p in parked if B.suffix_task(p) in (None, eval_id)}, "the parked attempts' records are still listed"
    # the corrected record (the producer re-mirrors sha = launched_as.sha): the same parked lists, the row files from the rerun store at its tree
    fx, ad, bt, facts = _read(tmp_path / "fixed", seeds, eval_id, **dict(kw, instruct_row_sha=None))
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and facts["not_run"] == [] and "parked" not in facts, facts["refusals"]
    r = _it_runs(bt)[0]
    assert r["extra"]["store_suffix_legacy"] == live and r["code_sha"] == Z.TREE2 and r["extra"]["served_tree_sha"] == Z.TREE2
    ls = r["extra"]["launch_set"]
    assert ls["sha"] == Z.TREE2 == ls["launched_as"]["sha"] and ls["tree_bound"] == Z.TREE2 and ls["tree_source"] == "row sha" and "row_tree_mismatch" not in ls
    # relaunches.json's parked_suffix: the LIST form is recorded on the run; the bare-string form (llama's) is read for the status alone (the live rows carrying it re-emit byte for byte)
    assert ls["relaunch"]["sha"] == Z.TREE2 and ls["relaunch"].get("parked_suffix") == (None if kw.get("instruct_parked_scalar") else parked)
    assert ls["relaunched_from"]["suffix"] == parked[0] and ls["relaunched_from"]["suffixes"] == parked
    listed = {g["suffix"]: g["verdict"] for g in facts["gate_records_without_store"]}
    want = {p: q.get("verdict", "FAIL") for p, q in zip(parked, kw["instruct_parked"]) if B.suffix_task(p) in (None, eval_id)}
    assert listed == want and all(g["parked_by_relaunches"] and g["named_by_relaunched_from"] for g in facts["gate_records_without_store"])
    c = [c for c in bt["run_contrast"] if c["run_id"] == r["run_id"]][0]
    assert c["anchor_run_id"] == Y.run_at(bt, TAG, NONE)["run_id"] and c["n_pairs"] == Z.N_FULL
    assert c["anchor_kind"] == "sibling_none" and c["extra"]["cross_standup"] is True, "the live shapes' corrected row pairs with another standup's none by alias"
    assert Y.run_at(bt, TAG, RELU)["extra"]["launch_set"]["tree_source"] == "primary tree"


def test_a_launched_as_sha_the_store_or_the_relaunch_record_disagrees_with_refuses_by_name(tmp_path, seeds):
    # launched_as says TREE3 while the row's sha (and the store) say TREE2 -- the inconsistent record refuses by name, nothing read
    fx, ad, bt, facts = _read(tmp_path / "la", seeds, "bigcodebench", instruct_relaunched=True, instruct_launched_as_sha=Z.TREE3)
    why = Z.refused(facts, IT_TAG, IT_KEY)
    assert f"the row's sha {Z.TREE2[:12]} is not the launched store's tree {Z.TREE3[:12]}" in why and "the record is inconsistent" in why and facts["runs"] == 5 and not _it_runs(bt)
    # relaunches.json names another tree than launched_as / the store
    fx, ad, bt, facts = _read(tmp_path / "rl", seeds, "bigcodebench", instruct_relaunched=True, instruct_relaunch_entry_sha=Z.TREE3)
    why = Z.refused(facts, IT_TAG, IT_KEY)
    assert f"{B.RELAUNCHES_FILE}[{IT_TAG}/{IT_KEY}] says tag 'r2' at tree '{Z.TREE3[:12]}'" in why and "the row's launched_as sha, which the entry must name" in why and facts["runs"] == 5
    # a launched_as sha that is no tree
    fx, ad, bt, facts = _read(tmp_path / "hex", seeds, "bigcodebench", instruct_relaunched=True, instruct_launched_as_sha="deadbeef")
    assert "launched_as names sha 'deadbeef', not a 40-hex producer tree" in Z.refused(facts, IT_TAG, IT_KEY)


def test_a_store_the_row_names_that_the_reader_did_not_read_lands_in_the_refusals_by_name(tmp_path, seeds):
    # (a) the rerun store sits under a suffix outside the grammar (`-bcb-r2`), and the row's launched_as names it: refused by name twice over -- as an unread store
    # directory and as the launched_as store -- never silently dropped, never not-run
    fx, ad, bt, facts = _read(tmp_path / "grammar", seeds, "bigcodebench", instruct_relaunched=True, instruct_store_suffix_extra="-bcb-r2")
    why = Z.refused(facts, IT_TAG, IT_KEY)
    assert "outside the grammar" in why and facts["runs"] == 5 and facts["not_run"] == [] and not _it_runs(bt), facts["refusals"]
    assert any(f"bcbw-{IT_TAG}-{Z.TREE2[:7]}-bcb-r2" in x for x in facts["refusals"])
    # (b) the rerun store is task-tagged for BigCodeBench and the BFCL row's launched_as names it: the BFCL source refuses by name (it is not its store) and the
    # BigCodeBench source reads it
    fx, ad, bt, facts = _read(tmp_path / "tagged", seeds, "bfcl", instruct_relaunched=True, instruct_store_suffix_extra="-r2-bcb")
    why = Z.refused(facts, IT_TAG, IT_KEY)
    assert "tagged for the task 'bigcodebench', not this source's (bfcl)" in why and facts["runs"] == 5 and facts["not_run"] == []
    fx, ad, bt, facts = _read(tmp_path / "tagged_bcb", seeds, "bigcodebench", instruct_relaunched=True, instruct_store_suffix_extra="-r2-bcb")
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and _it_runs(bt)[0]["extra"]["launch_set"]["task_tag"] == "bcb"
    # (c) launched_as names a store relaunched_from also parks: a contradictory declaration, refused by name
    def contradict(lr):
        lr["relaunched_from"]["suffixes"] = list(lr["relaunched_from"].get("suffixes") or [lr["relaunched_from"]["suffix"]]) + [lr["launched_as"]["suffix"]]
    fx, ad, bt, facts = _read(tmp_path / "contra", seeds, "bigcodebench", instruct_relaunched=True, instruct_row_override=contradict)
    assert "on launched_as AND parks it on relaunched_from" in Z.refused(facts, IT_TAG, IT_KEY) and facts["runs"] == 5 and facts["not_run"] == []
    # (d) launched_as names a store that is absent from the root: NOT RUN, and the not-run row names the store it is missing
    fx, ad, bt, facts = _read(tmp_path / "absent", seeds, "bigcodebench", instruct_relaunched=True, instruct_no_store=True)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and len(facts["not_run"]) == 1
    nr = facts["not_run"][0]
    assert (nr["tag"], nr["object_key"], nr["launched_as_suffix"], nr["launched_as_sha"], nr["store_dir_present"]) == (IT_TAG, IT_KEY, f"bcbw-{IT_TAG}-{Z.TREE2[:7]}-{Z.RELAUNCH_TAG}", Z.TREE2, False)
    assert nr["gate_verdict"] == "PASS", "the rerun's gate record is beside it; the store is what is missing"
