"""The `bigcodebench_full` reader (the contract quoted in the module) on the SYNTHETIC full-suite mirror of
`_bcbw_full_fixtures`: the gemma4e4b family's `s1` row (the `none`, a fitted ReLU-steer key, a prompted key) and its `s1p` row (the composed
difference_relu object and its own `none`), every store carrying BOTH full tasks over a SMALL pin (N_FULL = 5 ids), the gate records, the launch set
(rows with `item_set`), the launched pins / roster, `records/item_sets.json`, the two mirrored pin files and the three-number records written from the
same truth. What it pins beyond the shared reading (the appendix and seedrep tests):

* THE ITEM SET IS THE MIRROR'S, NEVER A LITERAL: the count comes from `records/item_sets.json` = the pin's `n` = `of` = the ids listed (5 here; 1,140 on
  the real mirror), the registry's declaration (`item_sets.yaml` `n_items` / `items_sha256`) must agree with it -- the ID -> expected-count table -- and
  every store attests it (`manifest.tinycap.bcb_full_item_set` = `{seed: null, n, of, ids_sha256}` with the digest `contracts.ids_sha256(pin["ids"])`
  mirrored as `_bcbw_common.pin_ids_sha256`; the label `full` on gen_stats / the summary / the outcome record; the launch-set row's `item_set {id,
  sha256, n}` with the pin FILE's sha256; the outcome record's count);
* THE NEW INSTRUMENT: every row files on `bigcodebench_full_protocol_greedy_4096` bound to the served model, `run.item_set_id bigcodebench_full_1140`,
  `extra.item_set` the attested block; run ids disjoint from the 200-item readers' over the same producer keys; the seeds declare the set with the
  contract's digest and count;
* REFUSED BY NAME: an ids_sha256 the manifest does not reproduce; a manifest `of` or `seed` that is not the pin's; a pin file whose sha256 is not the one
  item_sets.json names (every store); an absent item_sets.json or pin file (every store); item_sets.json saying n 1,140 over a 5-id pin (every store); the
  registry's declaration disagreeing with the pin (every store); a `pinned200` label where `full` is stamped; an outcome record short of n; a launch-set
  row without an `item_set` block or naming another; an item_sets.json without the set, and the id-keyed mapping layout reading like the list;
* the source state names the declared sets; `status_extra.item_set` names the attested one; registration, source, owner, version 1, every family of the
  model-key table; the 200-item specs carry none of this (their emission is pinned in `test_registry_adapter_bcbw_emission_pin.py`).
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

import _b1_fixtures as F
import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bigcodebench_allmodels as BCA
from sisteer.registry.adapters import bigcodebench_full as M
from sisteer.registry.adapters import bigcodebench_seedrep as SR
from sisteer.registry.adapters import lineage_common as LC

EVAL = "bigcodebench"
MK, TAG, TAG_P, SEL, SEL_P = Z.MK, Z.TAG, Z.TAG_P, Z.SEL, Z.SEL_P
NONE, RELU, PROMPTED, COMPOSED = Z.NONE, Z.RELU, Z.PROMPTED, Z.COMPOSED
N = Z.N_FULL


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _mirror(tmp_path, seeds, **kw):
    return Z.mirror(tmp_path, seeds, eval_id=EVAL, **kw)


def _adapter(fx, seeds, **kw) -> M.BigcodebenchFullAdapter:
    return Z.adapter(M.BigcodebenchFullAdapter, fx, seeds, **kw)


def _all_refused(facts, n: int, needle: str) -> None:
    assert facts["runs"] == 0 and facts["n_refusals"] == n, (facts["runs"], facts["refusals"])
    assert all(needle in x for x in facts["refusals"]), (needle, facts["refusals"])


# --------------------------------------------------------------------------------------------
# the spec, the seeds, the digest device
# --------------------------------------------------------------------------------------------

def test_the_spec_binds_the_full_suite_spellings_and_the_200_item_specs_carry_none_of_it():
    assert (M.SOURCE, M.NAME, M.EVAL_ID, M.TASK, M.RECORD_TASK, M.OUTCOMES_FILE) == ("bigcodebench_full",) * 2 + ("bigcodebench", "inspect/bigcodebench_full", "bigcodebench_full", "inspect__bigcodebench_full.json")
    assert (M.INSTRUMENT_SEED, M.ITEM_SET_ID, M.ITEM_SET_LABEL, M.ITEM_SET_KEY, M.PIN_FILE) == ("bigcodebench_full_protocol_greedy_4096", "bigcodebench_full_1140", "full", "bcb_full_item_set", "bcb_full.json")
    assert M.THREE_NUMBERS_FILE == "three_numbers_bigcodebench_full.json" and M.CAP == 4096 == SR.CAP and M.PROTOCOL_TEMPERATURE == 0.0 and M.LADDER == (4096,)
    s = M.SPEC
    assert s.full_item_set and s.n_items is None and s.item_set_id == M.ITEM_SET_ID and s.item_set_label == "full" and s.item_set_pin_file == "bcb_full.json"
    assert s.launch_set_eval is None and s.launch_set_eval_key == "bigcodebench" and s.models == tuple(B.MODEL_KEYS) and s.clamp_override is False and s.tool_calls_required is False
    assert s.roster_file == M.ROSTER_FILE == "bcb_bfcl_full_roster.yaml" != B.ROSTER_FILE and s.roster_variant_re.match("bcb_bfcl_full_roster.8ba0b48.yaml") and not s.roster_variant_re.match(B.ROSTER_FILE)
    assert SR.SPEC.roster_file == BCA.SPEC.roster_file == B.ROSTER_FILE == "bcb_bfcl_200_widen_roster.yaml" and SR.SPEC.roster_variant_re.pattern == B.ROSTER_VARIANT_RE.pattern
    assert s.recount_seed_provenance.startswith("none: a deterministic recount") and "bigcodebench_full_1140" in s.recount_seed_provenance and "200" not in s.recount_seed_provenance
    assert M.RECOUNT_ALGORITHM == "sisteer.registry.adapters.bigcodebench_full.recount" and M.recount_computation_id() != SR.recount_computation_id() != BCA.recount_computation_id()
    for spec in (SR.SPEC, BCA.SPEC):
        assert not spec.full_item_set and spec.n_items == 200 and spec.item_set_id is None and spec.item_set_label is None and spec.item_set_pin_file is None
        assert "200 attempted items" in spec.recount_seed_provenance
    assert M.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bigcodebench_full")) and M.ROOT_DEFAULT != BCA.ROOT_DEFAULT != SR.ROOT_DEFAULT
    assert B.ITEM_SETS_FILE == "item_sets.json" and B.ITEM_SET_LABEL_FULL == "full" and "contracts.ids_sha256" in B.ITEM_SET_RULE


def test_the_digest_device_is_contracts_ids_sha256_and_the_seeds_declare_the_contracts_set_and_a_new_instrument(seeds):
    from sisteer.capeval.tinycap import contracts as C
    ids = ["BigCodeBench/0", "BigCodeBench/1", "BigCodeBench/17"]
    assert B.pin_ids_sha256(ids) == C.ids_sha256(ids) == hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    assert B.pin_ids_sha256(ids) != hashlib.sha256("\n".join(ids).encode()).hexdigest(), "not cellstore's newline rule (mmlu_1531)"
    row = seeds.item_sets[M.ITEM_SET_ID]
    assert (row["eval_id"], row["n_items"], row["items_sha256"], row["subsample_rule"]) == ("bigcodebench", 1140, M.CONTRACT_IDS_SHA256, "full")
    assert row["pin_file_sha256"] == M.CONTRACT_PIN_FILE_SHA256 == "1aa5f1e4f6476dabad879e5fe8d5ad5eeaa01debd7f693a335d53d5651c4c9f2"
    assert row["items_sha256"] == "c309c92f137762955c16f8546e0ff9b1e84577f854fbba022904e493e3198a07" and "contracts.ids_sha256" in row["items_sha256_rule"]
    for mid in ("gemma4e4b_it", "llama8b_it", "nemotron3nano4b", "qwen35_4b_it"):
        inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id=mid)
        old = seeds.instrument_row(SR.INSTRUMENT_SEED, SR.EVAL_ID, model_id=mid)
        assert inst["item_set_id"] == M.ITEM_SET_ID and old["item_set_id"] == "bigcodebench_200" and inst["instrument_id"] != old["instrument_id"], mid
        assert (inst["harness"], inst["scoring_rule"], inst["token_cap_tokens"], inst["decode_seed_rule"], inst["thinking_mode"]) == ("inspect_evals", "pass_at_1_sandbox", 4096, "none", "off")
    assert seeds.instruments[M.INSTRUMENT_SEED]["item_set_ids"] == [M.ITEM_SET_ID]
    # the paper's wildcard row first, then the pipeline-test models' EXPLICIT canonical rows (qwen35_35b_a3b and gemma4_31b
    # follow the paper's canonical instrument, each with its own `<model>|bigcodebench|off|protocol` policy row in the seeds; the pin
    # was closed on the wildcard alone)
    assert seeds.instruments[M.INSTRUMENT_SEED]["decode_policy_refs"] == ["*|bigcodebench|off|protocol", "qwen35_35b_a3b|bigcodebench|off|protocol", "gemma4_31b|bigcodebench|off|protocol"]
    # the seeds copy the fixture hands the adapter declares the fixture's set; the real seeds are untouched
    s2 = Z.seeds_for(seeds)
    assert s2.item_sets[M.ITEM_SET_ID]["n_items"] == N and s2.item_sets[M.ITEM_SET_ID]["items_sha256"] == Z.IDS_SHA[EVAL] and seeds.item_sets[M.ITEM_SET_ID]["n_items"] == 1140
    assert s2.instruments is seeds.instruments


# --------------------------------------------------------------------------------------------
# the attested read
# --------------------------------------------------------------------------------------------

def test_attested_stores_file_on_the_new_instrument_with_the_item_set_taken_from_the_mirror(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    names = [os.path.basename(i.uri) for i in inputs]
    assert {ad.comparison_group(i) for i in inputs} == {TAG} and names.count(M.OUTCOMES_FILE) == 15 and not any(n.startswith(("inspect__bfcl", "inspect__bigcodebench_200")) for n in names)
    assert f"{B.ITEM_SETS_FILE}#{TAG}" in names and f"{M.PIN_FILE}#{TAG}" in names and f"{BFF.PIN_FILE}#{TAG}" not in names, "the index and THIS reader's pin file are group-shared records"
    assert f"{M.THREE_NUMBERS_FILE}#{TAG}" in names
    bt, facts = Z.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and facts["n_not_run"] == 0, facts["refusals"]
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="gemma4e4b_it")
    entry = fx["entries"][0]
    for tag, okey in ((TAG, NONE), (TAG, RELU), (TAG, PROMPTED), (TAG_P, COMPOSED), (TAG_P, NONE)):
        r = Z.run_at(bt, tag, okey)
        assert r["item_set_id"] == M.ITEM_SET_ID and r["instrument_id"] == inst["instrument_id"] and r["lane_source_legacy"] == M.SOURCE and r["eval_id"] == "bigcodebench", (tag, okey)
        assert r["n_items_planned"] == N and r["n_items_generated"] == N and r["token_cap_tokens"] == 4096 and r["thinking_mode"] == "off" and r["decode_seed_rule"] == "none"
        blk = r["extra"]["item_set"]
        assert (blk["id"], blk["n"], blk["of"], blk["seed"], blk["ids_sha256"], blk["pin_file"], blk["pin_file_sha256"], blk["label"]) == (M.ITEM_SET_ID, N, N, None, Z.IDS_SHA[EVAL], M.PIN_FILE, fx["pin_shas"][EVAL], "full")
        assert blk["manifest_key"] == "bcb_full_item_set" and blk["pin_task"] == "bigcodebench_full" and blk["registry_n_items"] == N and blk["registry_items_sha256"] == Z.IDS_SHA[EVAL]
        assert blk["launch_set_item_set"] == {"id": M.ITEM_SET_ID, "sha256": fx["pin_shas"][EVAL], "n": N} and blk["rule"] == B.ITEM_SET_RULE and "canonical JSON" in blk["ids_sha256_rule"]
        assert blk["index_record"].endswith(f"records/{B.ITEM_SETS_FILE}") and blk["pin_record"].endswith(f"records/{M.PIN_FILE}")
        assert r["extra"]["item_set_pin"] == {"ids_sha256": Z.IDS_SHA[EVAL], "n": N, "of": N, "seed": None}, "the manifest's block, recorded as on the 200-item rows"
        assert r["extra"]["three_numbers"]["n_attempted"] == N and r["extra"]["generation_state"].startswith("generated")
    assert entry["sha256"] == fx["pin_shas"][EVAL] == hashlib.sha256(open(os.path.join(fx["root"], B.RECORDS_DIR, M.PIN_FILE), "rb").read()).hexdigest(), "the index carries the pin FILE's bytes"
    # the three numbers over 5 items, the bound over n, the at-cap share over n
    r = Z.run_of(bt, RELU)
    c = Z.counts(fx["specs"][RELU])
    value, bound = Z.metrics_of(bt, r["run_id"], M.recount_computation_id())
    assert (value["k"], value["n"], value["value"]) == (c["correct"], c["n_scored"], c["correct"] / c["n_scored"]) == (2, 4, 0.5) and (bound["k"], bound["n"], bound["value"]) == (2, N, 0.4)
    assert bound["ci_hi"] == (c["correct"] + len(c["at_cap_unscored"])) / N == 0.6 and value["computation_id"] == I.computation_id(M.RECOUNT_ALGORITHM, M.RECOUNT_VERSION, M.RECOUNT_SEED_PROVENANCE)
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[M.RAW_AT_CAP_MEASURE]["numerator"], deg[M.RAW_AT_CAP_MEASURE]["denominator"]) == (2, N) and deg["rep_ratio_loop"]["measurement_state"] == "not_measured"
    assert len([x for x in bt["item_result"] if x["run_id"] == r["run_id"]]) == N and r["n_items_judged"] == 4 and f"/{N} at the cap" in r["notes"] and f"correct/{N} = 0.400" in r["notes"]
    # the anchors: the seed rows on the seed none, the composed row on the paired tag's own none; the contrast text names the count
    contrasts = {cn["run_id"]: cn for cn in bt["run_contrast"]}
    n = Z.run_at(bt, TAG, NONE)
    assert {contrasts[Z.run_of(bt, k)["run_id"]]["anchor_run_id"] for k in (RELU, PROMPTED)} == {n["run_id"]} and contrasts[Z.run_of(bt, COMPOSED)["run_id"]]["anchor_run_id"] == Z.run_at(bt, TAG_P, NONE)["run_id"]
    assert f"the same {N} items" in contrasts[r["run_id"]]["extra"]["method_text"]
    # status: the attested set; the source state: the declared sets
    assert facts["item_set"]["id"] == M.ITEM_SET_ID and facts["item_set"]["n"] == N and facts["item_set"]["problem"] is None and facts["item_set_rule"] == B.ITEM_SET_RULE
    states = {os.path.basename(s.input_uri).rsplit("#", 1)[0]: s for s in ad.source_states(X.ctx(), inputs)}
    assert states[B.ITEM_SETS_FILE].current_members == [BFF.ITEM_SET_ID, M.ITEM_SET_ID] and states[B.ITEM_SETS_FILE].json_path == "$"
    assert len(states[B.LAUNCH_SET_FILE].current_members) == 5 and states[B.LAUNCH_SET_FILE].json_path == "$.evals.bigcodebench.launch"
    assert "run_state" not in bt


def test_the_item_sets_index_may_be_the_id_keyed_mapping_and_the_full_rosters_per_tree_copy_is_discovered(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, item_sets_layout="dict", roster_variant=True)
    ad = _adapter(fx, seeds)
    names = [os.path.basename(i.uri) for i in ad.discover()]
    assert f"{Z.ROSTER_FILE}#{TAG}" in names and f"bcb_bfcl_full_roster.{Z.TREE[:7]}.yaml#{TAG}" in names and not any(n.startswith(B.ROSTER_FILE) for n in names)
    bt, facts = Z.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    assert Z.run_at(bt, TAG, NONE)["extra"]["launched_roster"]["roster_record"].endswith(f"records/{Z.ROSTER_FILE_T1}"), "the per-tree copy the index names for the store's hash"
    assert Z.run_at(bt, TAG, NONE)["extra"]["item_set"]["pin_file_sha256"] == fx["pin_shas"][EVAL]
    assert B.item_set_entries([{"id": "a", "sha256": "x"}]) == [{"id": "a", "sha256": "x"}] and B.item_set_entries({"item_sets": {"a": {"sha256": "x"}}}) == [{"sha256": "x", "id": "a"}]
    assert B.item_set_entries({"schema": "s", "a": {"sha256": "x"}, "b": {"n": 1}}) == [{"sha256": "x", "id": "a"}] and B.item_set_entries("nope") == []


def test_the_two_full_readers_and_the_200_item_readers_mint_distinct_run_ids_over_the_same_producer_keys(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, _f = Z.read(_adapter(fx, seeds))
    fx_f = Z.mirror(tmp_path, seeds, eval_id="bfcl", root=str(tmp_path / "bfcl_full"))
    bt_f, _ff = Z.read(Z.adapter(BFF.BfclFullAdapter, fx_f, seeds))
    ids = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    ids_f = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt_f["run"]}
    inst200 = str(seeds.instrument_row(BCA.INSTRUMENT_SEED, BCA.EVAL_ID, model_id="gemma4e4b_it")["instrument_id"])
    for k, r in ids.items():
        assert r["run_id"] != ids_f[k]["run_id"] and r["producer_run_key_legacy"] == ids_f[k]["producer_run_key_legacy"] and r["item_set_id"] != ids_f[k]["item_set_id"] == "bfcl_full_3981"
        assert r["run_id"] != BCA.run_id_for(k[0], r["producer_run_key_legacy"].split("/", 1)[1], k[1], inst200), "another item set, another instrument, another run id over the same store"
        assert r["run_id"] == M.run_id_for(k[0], r["producer_run_key_legacy"].split("/", 1)[1], k[1], r["instrument_id"])


# --------------------------------------------------------------------------------------------
# refused by name
# --------------------------------------------------------------------------------------------

def test_a_manifest_digest_the_pin_does_not_reproduce_refuses_the_store_by_name(tmp_path, seeds):
    def wrong_digest(block):
        block["ids_sha256"] = hashlib.sha256(b"another id list").hexdigest()
    fx = _mirror(tmp_path, seeds, manifest_item_set_override=wrong_digest)
    bt, facts = Z.read(_adapter(fx, seeds))
    why = Z.refused(facts, TAG, NONE)
    assert "manifest.json:tinycap.bcb_full_item_set does not attest the pinned item set bigcodebench_full_1140" in why and f"ids_sha256 '{hashlib.sha256(b'another id list').hexdigest()}' (the pin: '{Z.IDS_SHA[EVAL]}')" in why
    assert facts["runs"] == 4 and {r["campaign_cell_id_legacy"] for r in bt["run"] if r["view_legacy"] == TAG} == {RELU, PROMPTED}, "the siblings file"
    # the seed `none` refused: the own pairing is gone (`anchor_refused`) and the anchor currency re-anchors the row on the paired tag's `none` (sibling_none)
    blk = Z.run_of(bt, RELU)["extra"]["anchor"]
    assert blk["own_anchor_state"] == B.ANCHOR_REFUSED and blk["state"] == "contrasted" and blk["reanchored"] is True and blk["anchor_view_legacy"] == TAG_P
    def wrong_of_and_seed(block):
        block["of"] = 1140
        block["seed"] = 0
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "of"), manifest_item_set_override=wrong_of_and_seed)
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    why2 = Z.refused(facts2, TAG, NONE)
    assert "seed 0 (the pin: None)" in why2 and f"of 1140 (the pin: {N})" in why2 and facts2["runs"] == 4
    def absent(block):
        block.clear()
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "absent"), manifest_item_set_override=absent)
    bt3, facts3 = Z.read(_adapter(fx3, seeds))
    assert "does not attest the pinned item set" in Z.refused(facts3, TAG, NONE) and facts3["runs"] == 4


def test_a_pin_file_whose_sha256_is_not_the_one_item_sets_json_names_refuses_every_store(tmp_path, seeds):
    def reorder(pin):
        pin["ids"] = list(reversed(pin["ids"]))
    fx = _mirror(tmp_path, seeds, pin_override=reorder, index_sha_stale=True)
    bt, facts = Z.read(_adapter(fx, seeds))
    _all_refused(facts, 5, f"the mirrored pin file records/{M.PIN_FILE} has sha256")
    assert f"{B.ITEM_SETS_FILE} says {fx['entries'][0]['sha256'][:12]}" in facts["refusals"][0] and facts["item_set"]["problem"] == facts["refusals"][0].split(" -- ", 1)[1]
    # the same edit with the index re-hashed passes the file check and fails on the digest the manifest attests (the ids moved, the digest with them)
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "rehashed"), pin_override=reorder)
    bt2, facts2 = Z.read(_adapter(fx2, seeds, registry_sha={EVAL: B.pin_ids_sha256(list(reversed(Z.PIN_IDS[EVAL])))}))
    assert facts2["runs"] == 0 and facts2["n_refusals"] == 5 and all("does not attest the pinned item set" in x and "ids_sha256" in x for x in facts2["refusals"]), facts2["refusals"]


def test_an_absent_item_sets_json_or_pin_file_refuses_every_store_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, no_item_sets=True)
    ad = _adapter(fx, seeds)
    assert not any(os.path.basename(i.uri).startswith(B.ITEM_SETS_FILE) for i in ad.discover())
    bt, facts = Z.read(ad)
    _all_refused(facts, 5, f"records/{B.ITEM_SETS_FILE} is absent beside the stores")
    assert "bigcodebench_full_1140" in facts["refusals"][0] and facts["item_set"]["problem"] and facts["item_set"]["n"] is None
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "nopin"), no_pin_file=True)
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    _all_refused(facts2, 5, f"records/{M.PIN_FILE} (the pin file {B.ITEM_SETS_FILE} names for bigcodebench_full_1140")
    assert "is not mirrored beside the stores" in facts2["refusals"][0]


def test_an_index_count_the_pin_does_not_carry_refuses_every_store_naming_both_numbers(tmp_path, seeds):
    """item_sets.json says n 1,140; the pin lists 5 ids: the count is never taken from the index alone, the two must agree."""
    def n_1140(entries):
        next(e for e in entries if e["id"] == M.ITEM_SET_ID)["n"] = 1140
    fx = _mirror(tmp_path, seeds, item_sets_override=n_1140)
    bt, facts = Z.read(_adapter(fx, seeds))
    _all_refused(facts, 5, f"{B.ITEM_SETS_FILE} says n 1140, the pin carries {N} ids")
    # the pin's own n / of disagreeing with its ids
    def n_of(pin):
        pin["n"] = 1140
        pin["of"] = 1140
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "pin_n"), pin_override=n_of)
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    _all_refused(facts2, 5, f"n 1140 is not the {N} ids it lists; of 1140 is not the {N} ids it lists")
    # the pin's task and seed
    def task_seed(pin):
        pin["task"] = "bigcodebench_200"
        pin["seed"] = 0
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "pin_task"), pin_override=task_seed)
    bt3, facts3 = Z.read(_adapter(fx3, seeds))
    _all_refused(facts3, 5, "seed 0 is not null")
    assert "task 'bigcodebench_200' is not 'bigcodebench_full'" in facts3["refusals"][0]
    # the index without the set, or naming another file
    def drop(entries):
        entries[:] = [e for e in entries if e["id"] != M.ITEM_SET_ID]
    fx4 = _mirror(tmp_path, seeds, root=str(tmp_path / "noentry"), item_sets_override=drop)
    bt4, facts4 = Z.read(_adapter(fx4, seeds))
    _all_refused(facts4, 5, f"{B.ITEM_SETS_FILE} names 0 entries for item set bigcodebench_full_1140")
    assert "it names: bfcl_full_3981" in facts4["refusals"][0]
    def other_file(entries):
        next(e for e in entries if e["id"] == M.ITEM_SET_ID)["file"] = "bcb_full.8ba0b48.json"
    fx5 = _mirror(tmp_path, seeds, root=str(tmp_path / "otherfile"), item_sets_override=other_file)
    bt5, facts5 = Z.read(_adapter(fx5, seeds))
    _all_refused(facts5, 5, "file 'bcb_full.8ba0b48.json' is not the pin file this reader reads (bcb_full.json)")


def test_the_registrys_declaration_disagreeing_with_the_mirrored_pin_refuses_every_store(tmp_path, seeds):
    """The ID -> expected-count table is item_sets.yaml: the real row says 1,140 and the real digest; a mirror pinning anything else refuses by name (here the
    fixture's 5-id pin against the REAL declaration, and against a declaration with another digest)."""
    fx = _mirror(tmp_path, seeds)
    bt, facts = Z.read(X.adapter(M.BigcodebenchFullAdapter, fx, seeds))      # the real seeds: n_items 1140, the contract's digest
    _all_refused(facts, 5, f"item_sets.yaml declares bigcodebench_full_1140 with n_items 1140; the mirrored pin records/{M.PIN_FILE} carries {N} ids")
    bt2, facts2 = Z.read(_adapter(fx, seeds, registry_sha={EVAL: hashlib.sha256(b"a declaration of other ids").hexdigest()}))
    _all_refused(facts2, 5, "item_sets.yaml declares bigcodebench_full_1140 with items_sha256")
    assert f"the mirrored pin's ids digest (contracts.ids_sha256) is {Z.IDS_SHA[EVAL][:12]}" in facts2["refusals"][0]


def test_a_pinned200_label_where_full_is_stamped_refuses_the_store(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, label_override="pinned200")
    bt, facts = Z.read(_adapter(fx, seeds))
    why = Z.refused(facts, TAG, NONE)
    assert "gen_stats.json:item_set is 'pinned200', not 'full'" in why and "not a full-suite pass of inspect/bigcodebench_full over bigcodebench_full_1140" in why and facts["runs"] == 4


def test_an_outcome_record_short_of_n_refuses_the_store_by_name(tmp_path, seeds):
    """No 'declared shortfall' shape exists in the contract, so none is read: the core's convention for an incomplete store (a refusal naming the counts)."""
    fx = _mirror(tmp_path, seeds, short_outcomes=("4",))
    bt, facts = Z.read(_adapter(fx, seeds))
    assert f"the outcome record holds {N - 1} items, not {N}" in Z.refused(facts, TAG, NONE) and facts["runs"] == 4


def test_a_launch_set_row_without_an_item_set_or_naming_another_refuses_the_store(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, launch_item_set_drop=(f"{SEL}/{NONE}",))
    bt, facts = Z.read(_adapter(fx, seeds))
    assert "the launch set's row carries no item_set block ({id, sha256, n}) naming bigcodebench_full_1140" in Z.refused(facts, TAG, NONE) and facts["runs"] == 4
    def other(block):
        block["id"] = "bigcodebench_200"
        block["n"] = 200
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "other"), launch_item_set_override=other)
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    assert facts2["runs"] == 0 and facts2["n_refusals"] == 5 and all("the launch set's row names item set (id 'bigcodebench_200'" in x and "n 200)" in x for x in facts2["refusals"]), facts2["refusals"]
    def other_sha(block):
        block["sha256"] = hashlib.sha256(b"another pin file").hexdigest()
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "othersha"), launch_item_set_override=other_sha)
    bt3, facts3 = Z.read(_adapter(fx3, seeds))
    assert facts3["runs"] == 0 and all(f"the mirrored set is (bigcodebench_full_1140, {fx3['pin_shas'][EVAL][:12]} the pin file's bytes, {N})" in x for x in facts3["refusals"])


def test_an_instrument_seed_binding_another_item_set_refuses_the_store(tmp_path, seeds, monkeypatch):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    monkeypatch.setattr(type(ad), "SPEC", M.SPEC.__class__(**{**M.SPEC.__dict__, "instrument_seed": SR.INSTRUMENT_SEED}))
    bt, facts = Z.read(ad)
    assert facts["runs"] == 0 and all("binds item set 'bigcodebench_200' for bigcodebench; this reader files the pinned set bigcodebench_full_1140" in x for x in facts["refusals"]), facts["refusals"]


def test_every_family_of_the_table_reads_and_a_not_run_row_files_nothing(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, qwen_store=True)
    ad = _adapter(fx, seeds)
    bt, facts, groups = Z.read_group(ad, Z.QWEN_TAG)
    assert groups == [TAG, Z.QWEN_TAG] and facts["n_refusals"] == 0 and facts["runs"] == 1 and bt["run"][0]["model_id"] == "qwen35_4b_it" and bt["run"][0]["item_set_id"] == M.ITEM_SET_ID
    assert BCA.SPEC.models == B.APPENDIX_MODEL_KEYS and M.SPEC.models == tuple(B.MODEL_KEYS), "the full sources bind every family; the appendix readers their three"
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "nr"), not_run=(NONE,))
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    assert facts2["n_refusals"] == 0 and facts2["runs"] == 4 and facts2["n_not_run"] == 1
    blk = Z.run_of(bt2, RELU)["extra"]["anchor"]
    assert blk["own_anchor_state"] == B.ANCHOR_NOT_YET_RUN and blk["state"] == "contrasted" and blk["reanchored"] is True, "the own pairing waits; the currency re-anchors on the paired none"


# --------------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------------

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


def test_runs_reach_ingested_on_the_new_item_set_and_the_derived_decode_policy_match_names_the_protocol_row(tmp_path, seeds):
    """Two passes (emission 4, the anchor currency): the first files every run and parks every comparison, the fold makes the `none` runs
    current and the decision input moves, the second re-reads the group and files the three contrasts."""
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    objects = [ad.resolve_object(RELU, SEL, pin_key=Z.STAGED, model_key=MK)[0], ad.resolve_object(COMPOSED, SEL_P, pin_key=Z.STAGED_P, model_key=MK)[0]]
    root = _registry_root(tmp_path, seeds, objects)
    rep0, ing0, ing0b = F.run_and_ingest(root, ad, now="2026-09-23T08:00:00.000Z")          # (before X.NOW: the passes' `now` must advance)
    assert rep0.groups_refused == 0 and ing0.refused + ing0b.refused == 0, ([r.error for r in rep0.results], ing0.refusals, ing0b.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)]
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep.groups_refused == 0 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, item_set_id, lifecycle_state, n_items_planned, decode_policy_match, decode_policy_id FROM run WHERE lane_source_legacy = ? ORDER BY 1, 2", (M.SOURCE,))
    assert rows == [(TAG, RELU, M.ITEM_SET_ID, "ingested", N, "exact", "gemma4e4b_it|bigcodebench|off|protocol@v1"),
                    (TAG, NONE, M.ITEM_SET_ID, "ingested", N, "exact", "gemma4e4b_it|bigcodebench|off|protocol@v1"),
                    (TAG, PROMPTED, M.ITEM_SET_ID, "ingested", N, "exact", "gemma4e4b_it|bigcodebench|off|protocol@v1"),
                    (TAG_P, COMPOSED, M.ITEM_SET_ID, "ingested", N, "exact", "gemma4e4b_it|bigcodebench|off|protocol@v1"),
                    (TAG_P, NONE, M.ITEM_SET_ID, "ingested", N, "exact", "gemma4e4b_it|bigcodebench|off|protocol@v1")], rows
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)] and F.q(root, "SELECT n_items FROM item_set WHERE item_set_id = ?", (M.ITEM_SET_ID,)) == [(1140,)], "the registry's row is the real declaration"
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(5 * N,)]     # (two revisions after the second pass; the CURRENT view holds each row once)
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(3,)]
    assert F.q(root, "SELECT COUNT(*) FROM v_source_status_current WHERE json_extract(extra_json, '$.item_set.id') = ?", (M.ITEM_SET_ID,)) == [(1,)]
    rep2, _i3, _i4 = F.run_and_ingest(root, ad, now="2026-09-24T03:00:00.000Z")
    assert rep2.groups_refreshed == 0 and F.q(root, "SELECT DISTINCT lifecycle_state FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [("ingested",)]


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_and_declares_its_source_owner_version_and_model_set():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[M.NAME]()
    assert isinstance(inst, M.BigcodebenchFullAdapter) and inst.covers_sources == (M.SOURCE,) and inst.owner == "Overflow" and M.SOURCE == "bigcodebench_full"
    assert AB.emission_version_of(M.BigcodebenchFullAdapter) == "4" and AB.emission_version_of(BCA.BigcodebenchAllmodelsAdapter) == "2" and AB.emission_version_of(SR.BigcodebenchSeedrepAdapter) == "3"
    assert inst.rng_scope == "none" and inst.roots == (M.ROOT_DEFAULT,) and inst.shared_files()[-2:] == (B.ITEM_SETS_FILE, M.PIN_FILE) and BCA.BigcodebenchAllmodelsAdapter().shared_files()[-1] == B.RELAUNCHES_FILE
    assert inst.shared_files()[:2] == ("launch_sets_full.json", "launch_sets.json") and BCA.BigcodebenchAllmodelsAdapter().shared_files()[0] == B.LAUNCH_SET_FILE
    assert M.run_id_for(TAG, "k", NONE, "i") == I.run_id(LC.CAMPAIGN, f"{TAG}/k", NONE, "bigcodebench", "i")
    assert not isinstance(getattr(B.BcbwSeedrepReader, "name", None), str)


# --------------------------------------------------------------------------------------------
# the addendum of 2026-09-24 02:31Z: a second producer tree, relaunched rows, the launch set's trees, pins by hash within the set
# --------------------------------------------------------------------------------------------

def test_the_producers_list_shaped_rosters_index_reads_like_the_mapping_and_names_each_rosters_pins_by_hash(tmp_path, seeds):
    """Every mirror since 2026-09-24 00:22Z writes `rosters.json:rosters` as a LIST of {roster_sha256, file, tree_sha, pins_file, pins_sha256, stores}; the
    rosters branch's fixtures write the sha-keyed mapping. Both read; the list's pins_file / pins_sha256 select and VERIFY the per-tree pins copy."""
    for layout in ("list", "mapping"):
        fx = _mirror(tmp_path, seeds, root=str(tmp_path / layout), rosters_layout=layout)
        bt, facts = Z.read(_adapter(fx, seeds))
        assert facts["n_refusals"] == 0 and facts["runs"] == 5, (layout, facts["refusals"])
        lr = Z.run_at(bt, TAG, RELU)["extra"]["launched_roster"]
        assert lr["roster_file"] == Z.ROSTER_FILE and lr["tree_sha"] == Z.TREE and facts["roster_set"][0]["problem"] is None, layout
        assert Z.run_at(bt, TAG, RELU)["extra"]["launched_pins_record"].endswith(Z.PINS_FILE_T1), "the index's pins_file names the per-tree copy"
        if layout == "list":
            assert (lr["pins_file"], lr["pins_sha256"]) == (Z.PINS_FILE_T1, fx["pins_sha256"])
        else:
            assert "pins_sha256" not in lr, "the mapping shape carries no pins hash"
    # a pins hash the index names that the mirrored copy does not have refuses the store by name (the copy is matched by hash, never by name alone)
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "stale"), pins_sha_stale=True)
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    assert facts2["runs"] == 0 and facts2["n_refusals"] == 5 and all(f"the launched pins copy records/{Z.PINS_FILE_T1} has sha256 {fx2['pins_sha256'][:12]}; rosters.json names pins_sha256" in x for x in facts2["refusals"]), facts2["refusals"]
    # with no index at all the plain roster and pins are the set, as before
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "noindex"), rosters_layout="none")
    bt3, facts3 = Z.read(_adapter(fx3, seeds))
    assert facts3["n_refusals"] == 0 and facts3["runs"] == 5 and "roster_set" not in facts3 and Z.run_at(bt3, TAG, RELU)["extra"]["launched_pins_record"].endswith(Z.PINS_FILE_T1)


def test_a_relaunched_row_resolves_to_its_launched_as_store_and_the_parked_first_store_is_never_read(tmp_path, seeds):
    """The addendum: the row's launched_as names the `-r2` store at the second tree (its own `sha`), relaunched_from names the parked INCOMPLETE first store,
    relaunches.json files the relaunch, the gate at the second tree compares against the first record (cross_tree); the two pins copies differ in key order
    alone and each roster names its own by hash; a row without `sha` binds to the set's primary tree."""
    fx = _mirror(tmp_path, seeds, relaunched=(RELU,), launch_set_name="launch_sets_full.json", relaunches_dir="bcbw_records_full")
    ad = _adapter(fx, seeds)
    names = sorted({os.path.basename(i.uri).split("#")[0] for i in ad.discover()})
    assert "launch_sets_full.json" in names and "launch_sets.json" not in names and B.RELAUNCHES_FILE in names and Z.PINS_FILE_T1 in names and Z.PINS_FILE_T2 in names
    parked_dir = os.path.join(fx["root"], B.STORES_HEAD, Z.ROW_DIR, RELU, B.THINKING_DIR, X.suffix_of(TAG, RELU, tree=Z.TREE))
    r2_suffix = X.suffix_of(TAG, RELU, tree=Z.TREE2, suffix_extra="-r2")
    assert os.path.isdir(parked_dir) and os.path.isdir(os.path.join(fx["root"], B.STORES_HEAD, Z.ROW_DIR, RELU, B.THINKING_DIR, r2_suffix))
    assert B.SUFFIX_RE.match(r2_suffix).group("relaunch") == "-r2" and B.SUFFIX_RE.match(X.suffix_of(TAG, RELU)).group("relaunch") is None
    bt, facts = Z.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    assert facts["parked"] == [{"tag": TAG, "object_key": RELU, "suffix": X.suffix_of(TAG, RELU, tree=Z.TREE), "store": B.rel_store(Z.ROW_DIR, RELU, X.suffix_of(TAG, RELU, tree=Z.TREE)),
                                "relaunched_to": r2_suffix, "why": "synthetic: the first launch parked INCOMPLETE (a transport fault), relaunched at the fixed tree", "rule": B.RELAUNCH_RULE}]
    assert facts["n_parked"] == 1 and facts["launch_set_trees"] == [{"sha": Z.TREE, "role": "primary"}, {"sha": Z.TREE2, "role": "relaunch"}] and facts["launch_set_primary_tree"] == Z.TREE
    assert "parked" not in Z.read(_adapter(_mirror(tmp_path, seeds, root=str(tmp_path / "plain")), seeds))[1], "a group without a relaunch mints no parked list (the 200 readers' status extra stays byte for byte)"
    assert facts["relaunches_record"].endswith(f"bcbw_records_full/{B.RELAUNCHES_FILE}") and facts["launch_set"].endswith("launch_sets_full.json")
    r = Z.run_of(bt, RELU)
    assert r["extra"]["store_suffix_legacy"] == r2_suffix and r["code_sha"] == Z.TREE2 and r["extra"]["store_uri"].endswith(r2_suffix) and not r["extra"]["store_uri"].endswith(X.suffix_of(TAG, RELU, tree=Z.TREE))
    ls = r["extra"]["launch_set"]
    assert (ls["sha"], ls["tree_bound"], ls["tree_source"]) == (Z.TREE2, Z.TREE2, "row sha") and ls["relaunched_from"]["suffix"] == X.suffix_of(TAG, RELU, tree=Z.TREE)
    assert (ls["relaunch"]["tag"], ls["relaunch"]["sha"], ls["relaunch"]["rule"]) == ("r2", Z.TREE2, B.RELAUNCH_RULE) and ls["relaunch"]["record"].endswith(f"bcbw_records_full/{B.RELAUNCHES_FILE}")
    assert r["extra"]["gate"]["cross_tree"] is True and r["extra"]["gate"]["compared_against"]["sha"] == Z.TREE and r["extra"]["gate"]["compared_against"]["ruling"] == "D298"
    assert r["extra"]["launched_roster"]["tree_sha"] == Z.TREE2 and r["extra"]["launched_roster"]["roster_file"] == Z.ROSTER_FILE_T2 and r["extra"]["launched_roster"]["pins_sha256"] == fx["pins_sha256_t2"]
    assert r["extra"]["launched_pins_record"].endswith(Z.PINS_FILE_T2) and r["extra"]["launched_pins_tree_sha"] == Z.TREE2 and fx["pins_sha256_t2"] != fx["pins_sha256"], "the second copy: another byte order, the same content, matched by hash"
    n = Z.run_at(bt, TAG, NONE)
    assert n["code_sha"] == Z.TREE and (n["extra"]["launch_set"]["tree_bound"], n["extra"]["launch_set"]["tree_source"]) == (Z.TREE, "primary tree") and "sha" not in n["extra"]["launch_set"]
    assert n["extra"]["launched_pins_record"].endswith(Z.PINS_FILE_T1) and "cross_tree" not in n["extra"]["gate"] and "relaunch" not in n["extra"]["launch_set"]
    # the contrast still pairs the relaunched row with its tag's none
    contrasts = {cn["run_id"]: cn for cn in bt["run_contrast"]}
    assert contrasts[r["run_id"]]["anchor_run_id"] == n["run_id"]
    # the same mirror with relaunches.json under records/ reads the same
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "recs"), relaunched=(RELU,))
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    assert facts2["n_refusals"] == 0 and facts2["runs"] == 5 and facts2["relaunches_record"].endswith(f"records/{B.RELAUNCHES_FILE}") and facts2["launch_set"].endswith("launch_sets.json")


def test_a_row_whose_tree_the_store_does_not_carry_refuses_by_name(tmp_path, seeds):
    # a row with `sha` absent binds to the PRIMARY tree: the relaunched store (TREE2) then refuses
    fx = _mirror(tmp_path, seeds, relaunched=(RELU,), row_sha_override={f"{SEL}/{RELU}": None})
    bt, facts = Z.read(_adapter(fx, seeds))
    why = Z.refused(facts, TAG, RELU)
    assert f"the launch set says the row was launched at tree {Z.TREE[:12]} (the set's primary tree, no sha on the row); the store's manifest names {Z.TREE2[:12]}" in why and facts["runs"] == 4
    # a row with its own `sha` naming another tree refuses too
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "sha"), relaunched=(RELU,), row_sha_override={f"{SEL}/{NONE}": Z.TREE2})
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    assert f"launched at tree {Z.TREE2[:12]} (its own sha); the store's manifest names {Z.TREE[:12]}" in Z.refused(facts2, TAG, NONE) and facts2["runs"] == 4
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "badsha"), relaunched=(RELU,), row_sha_override={f"{SEL}/{NONE}": "deadbeef"})
    bt3, facts3 = Z.read(_adapter(fx3, seeds))
    assert "names sha 'deadbeef', not a 40-hex producer tree" in Z.refused(facts3, TAG, NONE)
    # without a `trees` block and without `sha` (every earlier launch set) nothing binds and nothing changes
    fx4 = _mirror(tmp_path, seeds, root=str(tmp_path / "notrees"))
    bt4, facts4 = Z.read(_adapter(fx4, seeds))
    assert facts4["runs"] == 5 and "launch_set_trees" not in facts4 and "tree_bound" not in Z.run_at(bt4, TAG, NONE)["extra"]["launch_set"]


def test_a_relaunched_store_without_its_relaunch_record_or_with_another_refuses_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, relaunched=(RELU,), no_relaunches=True)
    bt, facts = Z.read(_adapter(fx, seeds))
    assert f"no {B.RELAUNCHES_FILE} beside the stores (records / bcbw_records_full) names the relaunch {TAG}/{RELU}" in Z.refused(facts, TAG, RELU) and facts["runs"] == 4 and facts["n_parked"] == 1
    def other_sha(entries):
        entries[f"{TAG}/{RELU}"]["sha"] = Z.TREE
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "othersha"), relaunched=(RELU,), relaunch_entry_override=other_sha)
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    assert f"{B.RELAUNCHES_FILE}[{TAG}/{RELU}] says tag 'r2' at tree '{Z.TREE[:12]}'; the store is r2 at {Z.TREE2[:12]}" in Z.refused(facts2, TAG, RELU)
    def drop(entries):
        entries.clear()
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "noentry"), relaunched=(RELU,), relaunch_entry_override=drop)
    bt3, facts3 = Z.read(_adapter(fx3, seeds))
    assert f"{B.RELAUNCHES_FILE} names no entry {TAG}/{RELU} for the relaunched store" in Z.refused(facts3, TAG, RELU)
    # a `-r2` store the launch set's row does not name on launched_as
    def unname(doc):
        for ev in doc["evals"].values():
            for row in ev["launch"]:
                if row["cell"] == RELU:
                    row.pop("launched_as", None)
                    row.pop("relaunched_from", None)
    fx4 = _mirror(tmp_path, seeds, root=str(tmp_path / "unnamed"), relaunched=(RELU,), launch_override=unname)
    bt4, facts4 = Z.read(_adapter(fx4, seeds))
    whys = sorted(x for x in facts4["refusals"] if x.startswith(f"{TAG}|{RELU} -- "))
    assert len(whys) == 2 and "parked" not in facts4, "nothing is parked without the row's relaunched_from: BOTH store dirs refuse by name"
    assert any("spells a relaunch (r2) but the launch set's row does not name it on launched_as" in x for x in whys) and any("RESULT's last line does not read result=COMPLETE" in x for x in whys), whys
    assert facts4["runs"] == 4


def test_the_launch_set_is_read_by_name_and_two_launch_sets_at_once_refuse_every_store(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, launch_set_name="launch_sets_full.json")
    bt, facts = Z.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and facts["launch_set"].endswith("launch_sets_full.json") and M.SPEC.launch_set_files == ("launch_sets_full.json", "launch_sets.json")
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "both"), launch_set_name="both")
    bt2, facts2 = Z.read(_adapter(fx2, seeds))
    assert facts2["runs"] == 0 and facts2["n_refusals"] == 5 and all("carries 2 launch sets (launch_sets_full.json, launch_sets.json); the row table must be ONE file" in x for x in facts2["refusals"])
    assert facts2["launch_set_problem"] and facts2["launch_set"] is None
    assert SR.SPEC.launch_set_files == BCA.SPEC.launch_set_files == (B.LAUNCH_SET_FILE,), "the 200-item readers read the one name they always read"
