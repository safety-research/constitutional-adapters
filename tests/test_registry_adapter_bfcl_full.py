"""The `bfcl_full` reader (the contract quoted in `adapters/bigcodebench_full.py`) on the SYNTHETIC full-suite mirror of
`_bcbw_full_fixtures`: the same store shape as the BigCodeBench full reader's (one store holds both full tasks), read from ITS OWN root for ITS OWN task.
What it pins beyond the shared reading (tested in full on the BigCodeBench side): the BFCL FULL instrument (`bfcl_full_protocol_T0001_1024`, item set
`bfcl_full_3981`) bound to the served model, its own source and eval, the attestation on the BFCL manifest key (`bfcl_full_item_set`), label and pin
(`records/bfcl_full.json`), the clamp override and the tool-call route REQUIRED on this task, the three numbers over the mirrored count, the seeds'
declaration with the contract's digest, and that the two full readers mint distinct run ids over the same producer keys; registration, version 1."""
from __future__ import annotations

import hashlib
import os

import pytest

import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
from sisteer.registry import identity as I
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import bfcl_allmodels as BFA
from sisteer.registry.adapters import bfcl_full as M
from sisteer.registry.adapters import bfcl_seedrep as SR
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import lineage_common as LC

EVAL = "bfcl"
TAG, TAG_P = Z.TAG, Z.TAG_P
NONE, RELU, PROMPTED, COMPOSED = Z.NONE, Z.RELU, Z.PROMPTED, Z.COMPOSED
N = Z.N_FULL


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _mirror(tmp_path, seeds, **kw):
    return Z.mirror(tmp_path, seeds, eval_id=EVAL, **kw)


def _adapter(fx, seeds, **kw) -> M.BfclFullAdapter:
    return Z.adapter(M.BfclFullAdapter, fx, seeds, **kw)


def test_the_bfcl_rows_file_their_own_full_instrument_source_item_set_and_three_numbers(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    names = [os.path.basename(i.uri) for i in ad.discover()]
    assert names.count(M.OUTCOMES_FILE) == 15 and not any(n.startswith("inspect__bigcodebench") for n in names) and f"{M.THREE_NUMBERS_FILE}#{TAG}" in names
    assert f"{M.PIN_FILE}#{TAG}" in names and f"{BCF.PIN_FILE}#{TAG}" not in names and M.THREE_NUMBERS_FILE == "three_numbers_bfcl_full.json"
    bt, facts = Z.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="gemma4e4b_it")
    r = Z.run_of(bt, RELU)
    assert r["lane_source_legacy"] == M.SOURCE == "bfcl_full" and r["eval_id"] == "bfcl" and r["instrument_id"] == inst["instrument_id"] and r["item_set_id"] == "bfcl_full_3981" == M.ITEM_SET_ID
    assert r["model_id"] == "gemma4e4b_it" and r["harness"] == "bfcl_eval" and r["scoring_rule"] == "ast_match" and r["token_cap_tokens"] == 1024 == M.CAP and r["thinking_mode"] == "off"
    assert r["decode_sent_json"] == {"temperature": 0.001, "top_p": 1.0, "top_k": 0} and r["decode_source"] == "greedy" and r["instrument_match"] == "match" and r["n_items_planned"] == N
    assert r["extra"]["decode_policy_ref_legacy"] == "gemma4e4b_it|bfcl|off|protocol@v1" and r["decode_seed_rule"] == "none" and r["decode_policy_match"] == "no_policy"
    assert r["extra"]["sampling_clamp_override"] == {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"} and r["extra"]["tool_calls"] is True
    blk = r["extra"]["item_set"]
    assert (blk["id"], blk["n"], blk["of"], blk["ids_sha256"], blk["pin_file"], blk["pin_file_sha256"], blk["label"], blk["manifest_key"], blk["pin_task"]) == \
        (M.ITEM_SET_ID, N, N, Z.IDS_SHA[EVAL], "bfcl_full.json", fx["pin_shas"][EVAL], "full", "bfcl_full_item_set", "bfcl_full")
    assert r["extra"]["item_set_pin"] == {"ids_sha256": Z.IDS_SHA[EVAL], "n": N, "of": N, "seed": None}
    c = Z.counts(fx["specs"][RELU])
    value, bound = Z.metrics_of(bt, r["run_id"], M.recount_computation_id())
    assert (value["k"], value["n"]) == (c["correct"], c["n_scored"]) == (3, N) and (bound["k"], bound["n"], bound["producer_metric_spelling"]) == (3, N, "bound")
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[M.RAW_AT_CAP_MEASURE]["numerator"], deg[M.RAW_AT_CAP_MEASURE]["denominator"]) == (1, N) and deg["rep_ratio_loop"]["measurement_state"] == "not_measured"
    assert value["computation_id"] == I.computation_id(M.RECOUNT_ALGORITHM, M.RECOUNT_VERSION, M.RECOUNT_SEED_PROVENANCE) and "bfcl_full_3981" in M.RECOUNT_SEED_PROVENANCE
    contrasts = {cn["run_id"]: cn for cn in bt["run_contrast"]}
    n = Z.run_at(bt, TAG, NONE)
    assert {contrasts[Z.run_of(bt, k)["run_id"]]["anchor_run_id"] for k in (RELU, PROMPTED)} == {n["run_id"]} and contrasts[Z.run_of(bt, COMPOSED)["run_id"]]["anchor_run_id"] == Z.run_at(bt, TAG_P, NONE)["run_id"]
    assert facts["item_set"]["id"] == M.ITEM_SET_ID and facts["item_set"]["pin_file"] == "bfcl_full.json"


def test_the_seeds_declare_the_contracts_bfcl_set_and_a_new_instrument(seeds):
    row = seeds.item_sets[M.ITEM_SET_ID]
    assert (row["eval_id"], row["n_items"], row["items_sha256"], row["pin_file_sha256"], row["subsample_rule"]) == ("bfcl", 3981, M.CONTRACT_IDS_SHA256, M.CONTRACT_PIN_FILE_SHA256, "full")
    assert M.CONTRACT_PIN_FILE_SHA256 == "47afe04e967f6f2284bef576f96179e88efd45004a5b6f81cbfb348530205663" and M.CONTRACT_IDS_SHA256 == "61d1ac167feced0746d7d8705e49b6577ecf7493f9b54e47020f0a5cb60e2a81"
    for mid in ("gemma4e4b_it", "nemotron3nano4b", "qwen35_4b_it"):
        inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id=mid)
        assert inst["item_set_id"] == M.ITEM_SET_ID and inst["instrument_id"] != seeds.instrument_row(SR.INSTRUMENT_SEED, SR.EVAL_ID, model_id=mid)["instrument_id"]
        assert (inst["harness"], inst["scoring_rule"], inst["token_cap_tokens"], inst["decode_seed_rule"]) == ("bfcl_eval", "ast_match", 1024, "none")
    assert (M.TASK, M.RECORD_TASK, M.OUTCOMES_FILE, M.ITEM_SET_KEY, M.PIN_FILE) == ("inspect/bfcl_full", "bfcl_full", "inspect__bfcl_full.json", "bfcl_full_item_set", "bfcl_full.json")
    assert M.SPEC.clamp_override is True and M.SPEC.tool_calls_required is True and M.SPEC.full_item_set and M.SPEC.n_items is None and M.SPEC.models == tuple(B.MODEL_KEYS)
    assert not BFA.SPEC.full_item_set and BFA.SPEC.n_items == 200 and BFA.INSTRUMENT_SEED != M.INSTRUMENT_SEED


def test_the_refusals_read_on_this_task_with_the_bfcl_spellings(tmp_path, seeds):
    def wrong_digest(block):
        block["ids_sha256"] = hashlib.sha256(b"other").hexdigest()
    fx = _mirror(tmp_path, seeds, manifest_item_set_override=wrong_digest)
    bt, facts = Z.read(_adapter(fx, seeds))
    assert "manifest.json:tinycap.bfcl_full_item_set does not attest the pinned item set bfcl_full_3981 (records/bfcl_full.json" in Z.refused(facts, TAG, NONE) and facts["runs"] == 4
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "label"), label_override="pinned200")
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    assert "not a full-suite pass of inspect/bfcl_full over bfcl_full_3981" in Z.refused(facts2, TAG, NONE)
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "noidx"), no_item_sets=True)
    bt3, facts3 = Z.read(_adapter(fx3, seeds))
    assert facts3["runs"] == 0 and all("bfcl_full_3981" in x and B.ITEM_SETS_FILE in x for x in facts3["refusals"]) and len(facts3["refusals"]) == 5
    # the registry's real declaration (3,981, the contract's digest) against the 5-id fixture: every store refuses by name
    bt4, facts4 = Z.read(X.adapter(M.BfclFullAdapter, _mirror(tmp_path, seeds, root=str(tmp_path / "real")), seeds))
    assert facts4["runs"] == 0 and all("item_sets.yaml declares bfcl_full_3981 with n_items 3981" in x for x in facts4["refusals"])
    # the tool-call route is REQUIRED here (the BigCodeBench full reader records it)
    import glob, json
    fx5 = _mirror(tmp_path, seeds, root=str(tmp_path / "notool"))
    p = glob.glob(os.path.join(fx5["root"], B.STORES_HEAD, Z.ROW_DIR, NONE, B.THINKING_DIR, "*", "cells", "*", "manifest.json"))[0]
    m = json.load(open(p)); m["tinycap"]["served"]["tool_calls"] = False; m["route"]["tool_calls"] = False
    X.write(p, m)
    bt5, facts5 = Z.read(_adapter(fx5, seeds))
    assert "do not attest the tool-call route" in Z.refused(facts5, TAG, NONE) and facts5["runs"] == 4


def test_the_two_full_readers_mint_distinct_run_ids_over_the_same_producer_keys_and_never_the_200_item_ones(tmp_path, seeds):
    fx_f = _mirror(tmp_path, seeds, root=str(tmp_path / "bfcl_full"))
    fx_c = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "bigcodebench_full"))
    bt_f, _ff = Z.read(_adapter(fx_f, seeds))
    bt_c, _fc = Z.read(Z.adapter(BCF.BigcodebenchFullAdapter, fx_c, seeds))
    ids_f = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt_f["run"]}
    ids_c = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt_c["run"]}
    assert set(ids_f) == set(ids_c) == {(TAG, NONE), (TAG, RELU), (TAG, PROMPTED), (TAG_P, COMPOSED), (TAG_P, NONE)}
    inst200 = str(seeds.instrument_row(BFA.INSTRUMENT_SEED, BFA.EVAL_ID, model_id="gemma4e4b_it")["instrument_id"])
    for k in ids_f:
        assert ids_f[k]["run_id"] != ids_c[k]["run_id"] and ids_f[k]["producer_run_key_legacy"] == ids_c[k]["producer_run_key_legacy"] and ids_f[k]["object_id"] == ids_c[k]["object_id"]
        assert ids_f[k]["instrument_id"] != ids_c[k]["instrument_id"] and ids_f[k]["eval_id"] == "bfcl" and ids_c[k]["eval_id"] == "bigcodebench"
        assert ids_f[k]["run_id"] != BFA.run_id_for(k[0], ids_f[k]["producer_run_key_legacy"].split("/", 1)[1], k[1], inst200)


def test_the_reader_is_registered_and_declares_its_source_owner_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[M.NAME]()
    assert isinstance(inst, M.BfclFullAdapter) and inst.covers_sources == (M.SOURCE,) and inst.owner == "Overflow" and M.SOURCE == "bfcl_full"
    assert AB.emission_version_of(M.BfclFullAdapter) == "4" and AB.emission_version_of(BFA.BfclAllmodelsAdapter) == "2" and AB.emission_version_of(SR.BfclSeedrepAdapter) == "3"
    assert inst.rng_scope == "none" and inst.roots == (M.ROOT_DEFAULT,) and M.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bfcl_full")) and M.ROOT_DEFAULT != BCF.ROOT_DEFAULT != BFA.ROOT_DEFAULT
    assert inst.shared_files()[-2:] == (B.ITEM_SETS_FILE, M.PIN_FILE) and M.EVAL_ID == "bfcl" and M.INSTRUMENT_SEED == "bfcl_full_protocol_T0001_1024" and M.LADDER == (1024,)
    assert M.PROTOCOL_TEMPERATURE == 0.001 and M.recount_computation_id() != SR.recount_computation_id() != BFA.recount_computation_id()
