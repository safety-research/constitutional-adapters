"""The record contract and validator: strictness refusals name the field and path;
unknown keys are parked; `unverified` needs a reason; secret-shaped keys are refused; sha dedupe is
idempotent; malformed is refused, dependency-blocked is queued and then ingested (reverse-order and
cross-producer delivery); a blocked record goes stale after 7 days; run_patch field-set rules (S7).
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL


def _v(rec, path="fixture"):
    return VAL.validate_record(rec, path, R.record_sha(rec))


def test_gsm8k_without_item_set_refused_naming_field_and_path():
    p = synth.run_payload("k", i=0)
    p["eval_id"] = "gsm8k"
    del p["item_set_id"]
    rec = synth.rec("run", p, synth.t(0))
    out = _v(rec, "records/c/r/000001-run.jsonl#1")
    assert isinstance(out, VAL.Malformed)
    assert out.field == "item_set_id"
    assert "gsm8k" in out.reason and "item_set_id" in out.reason
    assert out.path == "records/c/r/000001-run.jsonl#1"
    p2 = synth.run_payload("k", i=0)
    p2["eval_id"] = "gsm8k"
    del p2["instrument_id"]
    out2 = _v(synth.rec("run", p2, synth.t(0)))
    assert isinstance(out2, VAL.Malformed) and out2.field == "instrument_id"


def test_unknown_keys_parked_in_extra():
    p = synth.run_payload("k", i=0)
    p["a_new_field"] = 7
    p["extra"] = {"nested": {"x": 1}}
    out = _v(synth.rec("run", p, synth.t(0)))
    assert isinstance(out, VAL.Validated)
    assert out.extra == {"a_new_field": 7, "nested": {"x": 1}}
    assert "a_new_field" not in out.payload


def test_unverified_requires_reason():
    p = synth.run_payload("k", i=0)
    p["code_sha"] = "unverified"
    out = _v(synth.rec("run", p, synth.t(0)))
    assert isinstance(out, VAL.Malformed) and out.field == "code_sha" and "field_confidence_json" in out.reason
    p["field_confidence_json"] = {"code_sha": {"confidence": "unverified", "reason": "legacy store without a sha"}}
    assert isinstance(_v(synth.rec("run", p, synth.t(0))), VAL.Validated)
    p["field_confidence_json"] = {"code_sha": {"confidence": "unverified", "reason": ""}}
    assert isinstance(_v(synth.rec("run", p, synth.t(0))), VAL.Malformed)


@pytest.mark.parametrize("key", ["api_key", "API_KEY", "hf_token", "wandb_api_key", "password", "token", "client_secret", "private_key"])
def test_secret_shaped_keys_refused(key):
    p = synth.run_payload("k", i=0)
    p["extra"] = {key: "x"}
    out = _v(synth.rec("run", p, synth.t(0)))
    assert isinstance(out, VAL.Malformed) and "secret-shaped" in out.reason


@pytest.mark.parametrize("key", ["token_cap_tokens", "si_token_count", "response_tokens", "fit_corpus_n_tokens"])
def test_schema_token_columns_are_not_secret_shaped(key):
    assert not VAL.SECRET_KEY_RE.search(key)


def test_vocabulary_refusal_names_field_value_and_allowed_set():
    p = synth.run_payload("k", i=0)
    p["serving_path"] = "vllm"
    out = _v(synth.rec("run", p, synth.t(0)))
    assert isinstance(out, VAL.Malformed) and out.field == "serving_path"
    assert "'vllm'" in out.reason and "lora_native" in out.reason and "SERVING_PATHS" in out.reason


def test_envelope_refusals():
    rec = synth.rec("run", synth.run_payload("k"), synth.t(0))
    bad = copy.deepcopy(rec)
    bad["written_utc"] = "2026-01-01T00:00:00Z"
    assert _v(bad).field == "written_utc"
    bad = copy.deepcopy(rec)
    bad["producer"]["substrate"] = "cloud"
    assert _v(bad).field == "producer.substrate"
    bad = copy.deepcopy(rec)
    bad["record_type"] = "nope"
    assert _v(bad).field == "record_type"
    bad = copy.deepcopy(rec)
    bad["producer"]["code_sha"] = "abc"
    assert _v(bad).field == "producer.code_sha"


def test_partial_item_record_refused_naming_missing_columns():
    rec = synth.item_records("r", "", 0, 1)[0]
    del rec["payload"]["created_utc"]
    del rec["payload"]["label"]
    out = _v(rec)
    assert isinstance(out, VAL.Malformed)
    assert "created_utc" in out.field and "label" in out.field and "whole row" in out.reason


def test_producer_may_not_claim_ingested():
    p = synth.run_payload("k", i=0)
    p["lifecycle_state"] = "ingested"
    assert isinstance(_v(synth.rec("run", p, synth.t(0))), VAL.Malformed)
    st = synth.rec("run_state", {"run_id": "r", "state": "ingested", "event_utc": synth.t(0), "evidence_kind": "harvester"}, synth.t(0))
    assert isinstance(_v(st), VAL.Malformed)


# ---- run_patch (S7) ------------------------------------------------------------------------------

def test_run_patch_allowed_fields_only():
    ok = synth.rec("run_patch", {"run_id": "r", "fields": {"node_hostname": "TESTHOST", "gpu_count": 1}}, synth.t(0))
    v = _v(ok)
    assert isinstance(v, VAL.Validated) and v.payload["fields"] == {"node_hostname": "TESTHOST", "gpu_count": 1}


def test_run_patch_identity_field_refused():
    bad = synth.rec("run_patch", {"run_id": "r", "fields": {"model_id": "other"}}, synth.t(0))
    out = _v(bad)
    assert isinstance(out, VAL.Malformed) and out.field == "model_id" and "identity" in out.reason
    bad2 = synth.rec("run_patch", {"run_id": "r", "fields": {"lifecycle_state": "judged"}}, synth.t(0))
    assert isinstance(_v(bad2), VAL.Malformed)


def test_run_patch_verified_field_in_fields_refused():
    bad = synth.rec("run_patch", {"run_id": "r", "fields": {"serving_path": "lens"}}, synth.t(0))
    out = _v(bad)
    assert isinstance(out, VAL.Malformed) and out.field == "serving_path" and "verify" in out.reason


def test_run_patch_unknown_field_refused_and_empty_refused():
    assert isinstance(_v(synth.rec("run_patch", {"run_id": "r", "fields": {"made_up": 1}}, synth.t(0))), VAL.Malformed)
    assert isinstance(_v(synth.rec("run_patch", {"run_id": "r"}, synth.t(0))), VAL.Malformed)


def test_every_patchable_and_verified_field_is_a_run_column_and_disjoint():
    from sisteer.registry import schema_def as S
    cols = set(S.TABLE_BY_NAME["run"].column_names)
    assert set(R.RUN_PATCHABLE_FIELDS) <= cols and set(R.RUN_VERIFIED_FIELDS) <= cols
    assert not set(R.RUN_PATCHABLE_FIELDS) & set(R.RUN_VERIFIED_FIELDS)
    assert not set(R.RUN_PATCHABLE_FIELDS) & set(R.RUN_IDENTITY_FIELDS)
    assert set(R.RUN_DERIVED_FIELDS) <= cols
    assert not set(R.RUN_DERIVED_FIELDS) & (set(R.RUN_PATCHABLE_FIELDS) | set(R.RUN_VERIFIED_FIELDS) | set(R.RUN_IDENTITY_FIELDS))


def test_decode_fields_are_patchable_except_the_derived_pair():
    """The five sent / resolved decode measurements and the declared `decode_source` are patchable (a pipeline
    re-read of the resolved sampler corrects them after registration); the derived pair `decode_policy_id` / `decode_policy_match`
    -- rewritten by `decode_match.stamp_run_rows` every cycle -- is refused BY NAME, pointing at the fields it is derived from;
    `decode_seed_rule` stays identity and `instrument_match` stays patchable."""
    measured = ("decode_sent_json", "decode_resolved_json", "decode_temperature_sent", "decode_top_p_sent", "decode_top_k_sent", "decode_source")
    for f in measured + ("instrument_match",):
        assert f in R.RUN_PATCHABLE_FIELDS and f not in R.RUN_IDENTITY_FIELDS and f not in R.RUN_VERIFIED_FIELDS, f
    assert "decode_seed_rule" in R.RUN_IDENTITY_FIELDS and "decode_seed_rule" not in R.RUN_PATCHABLE_FIELDS
    assert R.RUN_DERIVED_FIELDS == ("decode_policy_id", "decode_policy_match")
    for f in R.RUN_DERIVED_FIELDS:
        assert f not in R.RUN_PATCHABLE_FIELDS and f not in R.RUN_IDENTITY_FIELDS
    # the plan's correction record validates whole, extra_json parked for the fold's additive merge
    fields = {"decode_sent_json": {"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "decode_temperature_sent": 1.0, "decode_top_p_sent": 0.95,
              "decode_top_k_sent": 20, "decode_resolved_json": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
              "decode_source": "generation_config", "extra_json": {"decision": "D635", "reason": "resolved sampler re-read", "source": "synthetic://standup.log"}}
    v = _v(synth.rec("run_patch", {"run_id": "r", "fields": fields}, synth.t(0)))
    assert isinstance(v, VAL.Validated), v
    assert set(v.payload["fields"]) == set(measured)
    assert v.payload["fields"]["decode_source"] == "generation_config" and v.payload["fields"]["decode_top_k_sent"] == 20
    assert json.loads(v.payload["fields"]["decode_sent_json"]) == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    assert v.extra == {"decision": "D635", "reason": "resolved sampler re-read", "source": "synthetic://standup.log"}
    # a decode_source outside the vocabulary is still refused by the column rule
    bad_src = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"decode_source": "guessed"}}, synth.t(0)))
    assert isinstance(bad_src, VAL.Malformed) and bad_src.field == "decode_source"
    # the derived pair: refused by name, the reason names the derivation and the fields to patch instead
    for f, val in (("decode_policy_id", "p_vendor"), ("decode_policy_match", "exact")):
        out = _v(synth.rec("run_patch", {"run_id": "r", "fields": {f: val, "job_name": "x"}}, synth.t(0)))
        assert isinstance(out, VAL.Malformed) and out.field == f, out
        assert "derived" in out.reason and "stamp_run_rows" in out.reason and "decode_sent_json" in out.reason and "decode_source" in out.reason
    # the seed rule is identity: refused as such, not as 'not patchable'
    out = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"decode_seed_rule": "per_item"}}, synth.t(0)))
    assert isinstance(out, VAL.Malformed) and out.field == "decode_seed_rule" and "identity" in out.reason


def test_n_items_planned_is_patchable_and_a_non_integer_is_refused_by_name():
    """The plan's item count joins RUN_PATCHABLE_FIELDS (it is in neither RUN_VERIFIED_FIELDS nor
    RUN_IDENTITY_FIELDS); the column's INTEGER shape still holds at the validator."""
    assert "n_items_planned" in R.RUN_PATCHABLE_FIELDS
    assert "n_items_planned" not in R.RUN_VERIFIED_FIELDS and "n_items_planned" not in R.RUN_IDENTITY_FIELDS
    ok = synth.rec("run_patch", {"run_id": "r", "fields": {"n_items_planned": 10}}, synth.t(0))
    v = _v(ok)
    assert isinstance(v, VAL.Validated) and v.payload["fields"] == {"n_items_planned": 10}
    bad = synth.rec("run_patch", {"run_id": "r", "fields": {"n_items_planned": "10"}}, synth.t(0))
    out = _v(bad)
    assert isinstance(out, VAL.Malformed) and out.field == "n_items_planned" and "must be an integer" in out.reason
    assert isinstance(_v(synth.rec("run_patch", {"run_id": "r", "fields": {"n_items_planned": 10.0}}, synth.t(0))), VAL.Malformed)


def test_multi_turn_steering_scope_is_patchable_and_the_vocabulary_still_holds_at_the_validator():
    """The scope is derived from the applied-positions mode (patchable: the standup's endpoint record stamps the served mode) and the
    eval's category, so it follows the mode into RUN_PATCHABLE_FIELDS -- `registry reconcile multi-turn-scope` corrects the job-start
    default by run_patch; it is in neither RUN_IDENTITY_FIELDS nor RUN_VERIFIED_FIELDS and no part of `identity.run_id`. The column's
    CHECK vocabulary still holds at the validator, and `non_template` is an applied-positions mode a patch may stamp."""
    assert "multi_turn_steering_scope" in R.RUN_PATCHABLE_FIELDS
    assert "multi_turn_steering_scope" not in R.RUN_IDENTITY_FIELDS and "multi_turn_steering_scope" not in R.RUN_VERIFIED_FIELDS
    ok = synth.rec("run_patch", {"run_id": "r", "fields": {"multi_turn_steering_scope": "every_turn_prefill_and_decode",
                                                           "applied_positions_mode": "non_template"}}, synth.t(0))
    v = _v(ok)
    assert isinstance(v, VAL.Validated) and v.payload["fields"] == {"multi_turn_steering_scope": "every_turn_prefill_and_decode",
                                                                   "applied_positions_mode": "non_template"}
    bad = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"multi_turn_steering_scope": "every_other_turn"}}, synth.t(0)))
    assert isinstance(bad, VAL.Malformed) and bad.field == "multi_turn_steering_scope"
    bad_mode = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"applied_positions_mode": "template_only"}}, synth.t(0)))
    assert isinstance(bad_mode, VAL.Malformed) and bad_mode.field == "applied_positions_mode"


# ---- judgment identity ----------------------------------------------------------------------

def test_a_judgment_record_id_must_derive_from_its_own_run_and_judge_under_one_of_the_forms_in_use():
    """The pure rule (validate._judgment_id_refusal / identity.judgment_id_form): the plain form, the claim form and the producer form
    validate; an id minted elsewhere, or a same-id record naming another run or judge, is refused naming `judgment_id` and the forms
    tried; a historical record is admitted by sha alone (GRANDFATHERED_RECORD_SHAS), never by shape."""
    rid = "synthcampaign:syntheval:aaaaaaaaaaaa:bbbbbbbbbbbbbbbb"
    judge = synth.JUDGE_ID
    base = {"run_id": rid, "judge_id": judge, "status": "complete", "n_items_judged": 4}
    plain = synth.rec("judgment", {**base, "judgment_id": I.judgment_id(rid, judge)}, synth.t(1))
    claimed = synth.rec("judgment", {**base, "judgment_id": I.judgment_id(rid, judge, "s", synth.t(0)), "claimed_by_session": "s",
                                     "claimed_utc": synth.t(0)}, synth.t(1))
    produced = synth.rec("judgment", {**base, "judgment_id": I.judgment_id(rid, judge, "anchor"), "producer_session": "anchor"}, synth.t(1))
    hook = synth.rec("judgment", {**base, "judgment_id": I.judgment_id(rid, judge, "s", synth.t(0)), "claimed_by_session": "s",
                                  "claimed_utc": synth.t(0), "producer_session": "s"}, synth.t(1))
    for rec, form in ((plain, "run_judge"), (claimed, "claimed_by_session"), (produced, "producer_session"), (hook, "claimed_by_session")):
        assert isinstance(_v(rec), VAL.Validated), (form, _v(rec))
        assert I.judgment_id_form(rec["payload"]) == form
    assert I.judgment_identity_inputs(plain["payload"]) == {}
    assert I.judgment_identity_inputs(claimed["payload"]) == {"claimed_by_session": "s", "claimed_utc": synth.t(0)}
    assert I.judgment_identity_inputs(produced["payload"]) == {"producer_session": "anchor", "claimed_utc": None}
    # refused: a literal id, an id over another run, an id over another judge, a claim id stated without its inputs
    other_run = rid.replace("bbbbbbbb", "cccccccc")
    for payload in ({**base, "judgment_id": "j"},
                    {**base, "judgment_id": I.judgment_id(other_run, judge)},
                    {**base, "judgment_id": I.judgment_id(rid, synth.JUDGE_ID_2)},
                    {**base, "judgment_id": I.judgment_id(rid, judge, "s", synth.t(0))},
                    {**base, "judgment_id": I.judgment_id(rid, judge, "s", synth.t(0)), "claimed_by_session": "s", "claimed_utc": synth.t(5)}):
        out = _v(synth.rec("judgment", payload, synth.t(1)))
        assert isinstance(out, VAL.Malformed) and out.field == "judgment_id", payload
        assert "not derived from this record's run_id / judge_id" in out.reason and "(run_judge)" in out.reason and "claimed_by_session=" in out.reason
        assert I.judgment_id_form(payload) is None and I.judgment_identity_inputs(payload) is None
    # the grandfather: by sha, never by shape
    legacy = synth.rec("judgment", {**base, "judgment_id": "j"}, synth.t(1))
    sha = R.record_sha(legacy)
    assert isinstance(VAL.validate_record(legacy, "<p>", sha), VAL.Malformed)
    saved = VAL.GRANDFATHERED_RECORD_SHAS
    VAL.GRANDFATHERED_RECORD_SHAS = saved | {sha}
    try:
        assert isinstance(VAL.validate_record(legacy, "<p>", sha), VAL.Validated)
        assert isinstance(VAL.validate_record(legacy, "<p>", "0" * 64), VAL.Malformed)        # another sha of the same shape: refused
        assert isinstance(VAL.validate_record(legacy, "<p>"), VAL.Malformed)                  # no sha given: refused
    finally:
        VAL.GRANDFATHERED_RECORD_SHAS = saved
    assert sha not in VAL.GRANDFATHERED_RECORD_SHAS
    # the other column rules still come first: a status outside the vocabulary is named before the id
    out = _v(synth.rec("judgment", {**base, "judgment_id": "j", "status": "done"}, synth.t(1)))
    assert isinstance(out, VAL.Malformed) and out.field == "status"


# ---- the token-cap ladder facts of a run record -----------------------------

def test_a_run_record_stating_its_ladder_index_must_agree_with_its_ladder_and_its_cap():
    """Rule (i), pure (validate.run_cap_ladder_refusal): a `run` record that states `token_cap_rung_index` states a non-empty ladder of
    strictly ascending positive integers, an index inside it and a cap equal to the indexed level, else it is refused naming the field that
    disagrees; a record stating no index -- the legacy shapes -- is admitted; a historical record is admitted by sha alone
    (GRANDFATHERED_RECORD_SHAS), never by shape; none of the three fields is reachable by patch."""
    ladder = [4096, 8192, 16384]

    def run(**over):
        return synth.rec("run", synth.run_payload("k", i=0, **over), synth.t(0))

    ok = _v(run(token_cap_tokens=8192, token_cap_rung_index=1, token_cap_ladder_json=ladder))
    assert isinstance(ok, VAL.Validated), ok
    assert json.loads(ok.payload["token_cap_ladder_json"]) == ladder and ok.payload["token_cap_rung_index"] == 1 and ok.payload["token_cap_tokens"] == 8192
    # the serialized ladder is the same statement; the base level and the final level are levels like any other
    assert isinstance(_v(run(token_cap_tokens=4096, token_cap_rung_index=0, token_cap_ladder_json=json.dumps(ladder))), VAL.Validated)
    assert isinstance(_v(run(token_cap_tokens=16384, token_cap_rung_index=2, token_cap_ladder_json=ladder)), VAL.Validated)
    assert isinstance(_v(run(token_cap_tokens=512, token_cap_rung_index=0, token_cap_ladder_json=[512])), VAL.Validated)
    # the legacy shapes, admitted as they always were: a ladder with no index (the cap in it, or not), no ladder, an empty ladder
    for over in ({"token_cap_tokens": 8192, "token_cap_ladder_json": ladder}, {"token_cap_tokens": 1234, "token_cap_ladder_json": ladder},
                 {"token_cap_tokens": 8192}, {"token_cap_tokens": 8192, "token_cap_ladder_json": []}, {"token_cap_tokens": 8192, "token_cap_rung_index": None}):
        assert isinstance(_v(run(**over)), VAL.Validated), over
    # refused by name: the field that disagrees, the reason naming the rule and the values
    cases = [
        ({"token_cap_tokens": 8192, "token_cap_rung_index": 1}, "token_cap_ladder_json", "no token_cap_ladder_json"),
        ({"token_cap_tokens": 8192, "token_cap_rung_index": 1, "token_cap_ladder_json": []}, "token_cap_ladder_json", "no token_cap_ladder_json"),
        ({"token_cap_tokens": 8192, "token_cap_rung_index": 0, "token_cap_ladder_json": [8192, 4096]}, "token_cap_ladder_json", "strictly ascending"),
        ({"token_cap_tokens": 8192, "token_cap_rung_index": 0, "token_cap_ladder_json": [8192, 8192]}, "token_cap_ladder_json", "strictly ascending"),
        ({"token_cap_tokens": 4096, "token_cap_rung_index": 0, "token_cap_ladder_json": [4096, "8192"]}, "token_cap_ladder_json", "element 1 is '8192'"),
        ({"token_cap_tokens": 4096, "token_cap_rung_index": 1, "token_cap_ladder_json": [0, 4096]}, "token_cap_ladder_json", "element 0 is 0"),
        ({"token_cap_tokens": 4096, "token_cap_rung_index": 1, "token_cap_ladder_json": [True, 4096]}, "token_cap_ladder_json", "element 0 is True"),
        ({"token_cap_tokens": 8192, "token_cap_rung_index": 3, "token_cap_ladder_json": ladder}, "token_cap_rung_index", "outside"),
        ({"token_cap_tokens": 8192, "token_cap_rung_index": -1, "token_cap_ladder_json": ladder}, "token_cap_rung_index", "outside"),
        ({"token_cap_tokens": 8192, "token_cap_rung_index": 0, "token_cap_ladder_json": ladder}, "token_cap_tokens", "= 4096"),
        ({"token_cap_tokens": 16384, "token_cap_rung_index": 1, "token_cap_ladder_json": ladder}, "token_cap_tokens", "= 8192"),
    ]
    for over, field, needle in cases:
        out = _v(run(**over))
        assert isinstance(out, VAL.Malformed) and out.field == field, (over, out)
        assert needle in out.reason and "D648" in out.reason, (over, out.reason)
    # shape first: a non-integer index is the column rule's refusal, not this rule's
    out = _v(run(token_cap_tokens=8192, token_cap_rung_index="1", token_cap_ladder_json=ladder))
    assert isinstance(out, VAL.Malformed) and out.field == "token_cap_rung_index" and "must be an integer" in out.reason
    # the pure function on a live row's shapes (the ladder as the column's text), the way `registry cap-check` re-applies it
    assert VAL.run_cap_ladder_refusal(8192, None, "[]") is None and VAL.run_cap_ladder_refusal(8192, None, None) is None
    assert VAL.run_cap_ladder_refusal(8192, 1, "[4096,8192,16384]") is None
    assert VAL.run_cap_ladder_refusal(8192, 0, "[4096,8192,16384]")[0] == "token_cap_tokens"
    assert VAL.run_cap_ladder_refusal(8192, 0, "not json")[0] == "token_cap_ladder_json"
    assert VAL.run_cap_ladder_refusal(8192, 0, '{"a": 1}')[0] == "token_cap_ladder_json"
    assert VAL.run_cap_ladder_refusal(8192, True, "[8192]")[0] == "token_cap_rung_index"
    # the grandfather: by sha, never by shape
    legacy = run(token_cap_tokens=8192, token_cap_rung_index=0, token_cap_ladder_json=ladder)
    sha = R.record_sha(legacy)
    assert isinstance(VAL.validate_record(legacy, "<p>", sha), VAL.Malformed)
    saved = VAL.GRANDFATHERED_RECORD_SHAS
    VAL.GRANDFATHERED_RECORD_SHAS = saved | {sha}
    try:
        assert isinstance(VAL.validate_record(legacy, "<p>", sha), VAL.Validated)
        assert isinstance(VAL.validate_record(legacy, "<p>", "0" * 64), VAL.Malformed)
        assert isinstance(VAL.validate_record(legacy, "<p>"), VAL.Malformed)
    finally:
        VAL.GRANDFATHERED_RECORD_SHAS = saved
    assert sha not in VAL.GRANDFATHERED_RECORD_SHAS
    # the three fields are unreachable by patch: the cap is identity, the ladder fields are not patchable
    for f, val in (("token_cap_tokens", 4096), ("token_cap_rung_index", 0), ("token_cap_ladder_json", ladder)):
        assert f not in R.RUN_PATCHABLE_FIELDS
        out = _v(synth.rec("run_patch", {"run_id": "r", "fields": {f: val}}, synth.t(0)))
        assert isinstance(out, VAL.Malformed) and out.field == f, (f, out)
    assert "token_cap_tokens" in R.RUN_IDENTITY_FIELDS


def test_a_run_record_s_cap_ladder_facts_are_read_as_stated_so_a_boolean_index_is_refused_not_folded_as_its_integer():
    """`_normalise_payload` turns a boolean into its integer for every INTEGER column (the flag columns' convention), so a `run` record
    stating `token_cap_rung_index: true` with ladder [4096, 8192] and cap 8,192 reached the cap-ladder rule as index 1 -- agreeing with
    itself -- and folded a level-1 fact the producer never stated, which `registry cap-check`'s live-row re-check could not see. The rule
    reads the three facts as the payload states them: a boolean index, or a boolean cap beside a stated index, is refused naming its
    field; the integer form of the same record validates; a grandfathered sha is admitted as by every rule."""
    ladder = [4096, 8192]

    def run(**over):
        return synth.rec("run", synth.run_payload("k", i=0, **over), synth.t(0))

    out = _v(run(token_cap_tokens=8192, token_cap_rung_index=True, token_cap_ladder_json=ladder))
    assert isinstance(out, VAL.Malformed) and out.field == "token_cap_rung_index", out
    assert "token_cap_rung_index=True must be an integer" in out.reason and "D648" in out.reason, out.reason
    assert VAL.run_cap_ladder_refusal(8192, True, ladder) == (out.field, out.reason)       # the pure rule, the same verdict, end to end
    # `false` would have folded as level 0 (4,096 with a 4,096 cap agrees with itself too)
    out = _v(run(token_cap_tokens=4096, token_cap_rung_index=False, token_cap_ladder_json=ladder))
    assert isinstance(out, VAL.Malformed) and out.field == "token_cap_rung_index" and "token_cap_rung_index=False" in out.reason
    # a boolean cap beside a stated index: refused naming the cap, even where the indexed level is 1 (True == 1 in Python)
    out = _v(run(token_cap_tokens=True, token_cap_rung_index=0, token_cap_ladder_json=[1, 4096]))
    assert isinstance(out, VAL.Malformed) and out.field == "token_cap_tokens" and "token_cap_tokens=True" in out.reason and "D648" in out.reason, out
    assert VAL.run_cap_ladder_refusal(True, 0, [1, 4096])[0] == "token_cap_tokens" and VAL.run_cap_ladder_refusal(1, 0, [1, 4096]) is None
    # the same record in its integer form validates, the index as stated
    ok = _v(run(token_cap_tokens=8192, token_cap_rung_index=1, token_cap_ladder_json=ladder))
    assert isinstance(ok, VAL.Validated) and ok.payload["token_cap_rung_index"] == 1 and ok.payload["token_cap_tokens"] == 8192
    # the grandfather: by sha, as for every rule (the list gains nothing: 0 of the log's run records state a boolean in any cap field)
    legacy = run(token_cap_tokens=8192, token_cap_rung_index=True, token_cap_ladder_json=ladder)
    sha = R.record_sha(legacy)
    assert isinstance(VAL.validate_record(legacy, "<p>", sha), VAL.Malformed)
    saved = VAL.GRANDFATHERED_RECORD_SHAS
    VAL.GRANDFATHERED_RECORD_SHAS = saved | {sha}
    try:
        ok = VAL.validate_record(legacy, "<p>", sha)
        assert isinstance(ok, VAL.Validated) and ok.payload["token_cap_rung_index"] == 1     # admitted by sha; the normalizer's integer folds
        assert isinstance(VAL.validate_record(legacy, "<p>", "0" * 64), VAL.Malformed)
    finally:
        VAL.GRANDFATHERED_RECORD_SHAS = saved
    assert sha not in VAL.GRANDFATHERED_RECORD_SHAS


# ---- a run record's code_sha is never a placeholder -------------------------------------------------

def test_a_run_record_with_a_placeholder_code_sha_is_refused_by_name_unless_grandfathered():
    """Pure (validate.run_code_sha_refusal): a `run` record whose `code_sha` is forty identical hex digits -- the shape a test's
    synthetic producer block states -- is refused naming `code_sha`; a derived sha and the `unverified` form (with
    its field_confidence_json entry) are admitted; the column rule refuses a malformed sha first; a historical record is admitted by sha
    alone (GRANDFATHERED_RECORD_SHAS: the four records of the two placeholder-sha test runs), never by shape; the rule is payload-only --
    the envelope's producer.code_sha and other kinds' code_sha columns are not its business; and `code_sha` is identity, unreachable by
    patch."""

    def run(**over):
        return synth.rec("run", synth.run_payload("k", i=0, **over), synth.t(0))

    # the synthetic corpus itself conforms: its sha is derived, not a placeholder (synth.CODE_SHA == synth.code_sha("a"))
    assert isinstance(_v(run()), VAL.Validated)
    assert R.SHA1_RE.match(synth.CODE_SHA) and len(set(synth.CODE_SHA)) > 1 and synth.CODE_SHA == synth.code_sha("a")
    assert synth.code_sha("b") != synth.CODE_SHA and R.SHA1_RE.match(synth.code_sha("b")) and len(set(synth.code_sha("b"))) > 1
    # every placeholder digit is refused by name, the reason naming the rule and the value
    for d in "0123456789abcdef":
        out = _v(run(code_sha=d * 40))
        assert isinstance(out, VAL.Malformed) and out.field == "code_sha", (d, out)
        assert "placeholder" in out.reason and "D650" in out.reason and d * 40 in out.reason, out.reason
        assert VAL.run_code_sha_refusal(d * 40) == out.reason                   # the pure rule, the same verdict, end to end
    # admitted: a derived sha; `unverified` with its confidence entry (the column rule's business, untouched here); no value
    assert VAL.run_code_sha_refusal(synth.code_sha("z")) is None and VAL.run_code_sha_refusal("unverified") is None
    assert VAL.run_code_sha_refusal(None) is None
    fc = {"code_sha": {"confidence": "unverified", "reason": "git rev-parse HEAD unavailable in the registering checkout"}}
    assert isinstance(_v(run(code_sha="unverified", field_confidence_json=fc)), VAL.Validated)
    # shape first: a malformed sha is the column rule's refusal, not this rule's
    for bad in ("a" * 39, "A" * 40, "g" * 40, "a" * 41):
        out = _v(run(code_sha=bad))
        assert isinstance(out, VAL.Malformed) and out.field == "code_sha" and "does not match" in out.reason, (bad, out)
    out = _v(run(code_sha=42))
    assert isinstance(out, VAL.Malformed) and out.field == "code_sha" and "must be a string" in out.reason
    # payload-only: a placeholder in the ENVELOPE beside a derived payload sha validates -- the envelope check
    # admits any 40-hex -- and a `computation` record's code_sha column is not this rule's
    prod = synth.producer()
    prod["code_sha"] = "a" * 40
    assert isinstance(_v(synth.rec("run", synth.run_payload("k", i=0), synth.t(0), prod)), VAL.Validated)
    comp = copy.deepcopy(next(r for r in synth.seed_dimensions() if r["record_type"] == "computation"))
    assert comp["payload"]["code_sha"] == synth.CODE_SHA
    comp["payload"]["code_sha"] = "a" * 40
    assert isinstance(_v(comp), VAL.Validated)
    # the grandfather: by sha, never by shape; the list carries the four audited records
    legacy = run(code_sha="a" * 40)
    sha = R.record_sha(legacy)
    assert isinstance(VAL.validate_record(legacy, "<p>", sha), VAL.Malformed)
    saved = VAL.GRANDFATHERED_RECORD_SHAS
    VAL.GRANDFATHERED_RECORD_SHAS = saved | {sha}
    try:
        ok = VAL.validate_record(legacy, "<p>", sha)
        assert isinstance(ok, VAL.Validated) and ok.payload["code_sha"] == "a" * 40
        assert isinstance(VAL.validate_record(legacy, "<p>", "0" * 64), VAL.Malformed)
        assert isinstance(VAL.validate_record(legacy, "<p>"), VAL.Malformed)
    finally:
        VAL.GRANDFATHERED_RECORD_SHAS = saved
    assert sha not in VAL.GRANDFATHERED_RECORD_SHAS
    d650 = {"0028e3f5b5fc68e7335bfe3d9d946bc93de5f202aafb51a8fdcf3db766128309", "4dc25cca98b4a01b2eedbd49aec0a51e0acf225529650f300ca9c9269a7c6d4d",
            "98fe6d0d3da73bc9b918911dd20fb33ec76ccb16526399b87ce1bd8031f068f4", "b59823cdd3751a787992ac24ecd948a5b9196a46bf1a704be80dc06fb98d15b3"}
    assert d650 <= VAL.GRANDFATHERED_RECORD_SHAS
    # identity, unreachable by patch
    assert "code_sha" in R.RUN_IDENTITY_FIELDS and "code_sha" not in R.RUN_PATCHABLE_FIELDS
    out = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"code_sha": synth.code_sha("b")}}, synth.t(0)))
    assert isinstance(out, VAL.Malformed) and out.field == "code_sha"


# ---- harvester-level: dedupe, malformed vs blocked, stale ------------------------------------------

def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs, **kw):
    return R.write_records(os.path.join(root, "records", sub, name), recs, **kw)


def test_record_sha_dedupe_idempotent(tmp_path):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    _write(root, "dims", "000001-dims.jsonl", dims)
    r1 = H.ingest_once(root, workers=0, min_free=0)
    assert r1.ok - r1.arrival_records == len(dims) and r1.refused == 0
    # the same records again under ANOTHER file name → duplicates, nothing changes
    _write(root, "dims", "000002-dims-repush.jsonl", dims)
    r2 = H.ingest_once(root, workers=0, min_free=0)
    assert r2.ok - r2.arrival_records == 0 and r2.duplicates == len(dims)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM record_ingest WHERE file_uri NOT LIKE 'records/harvester/%'").fetchone()[0] == 2   # two FILES ranked
    assert conn.execute("SELECT COUNT(*) FROM model").fetchone()[0] == 2                  # no duplicated rows
    assert conn.execute("SELECT COUNT(*) FROM row_history WHERE table_name <> 'file_arrival'").fetchone()[0] == len(dims)   # one event per record, not two
    assert conn.execute("SELECT status FROM record_delivery WHERE file_uri LIKE '%repush%'").fetchone()[0] == "duplicate"
    conn.close()


def test_malformed_is_refused_not_blocked(tmp_path):
    root = _root(tmp_path)
    p = synth.run_payload("k", i=0)
    p["serving_path"] = "vllm"
    _write(root, "x", "000001-run.jsonl", [synth.rec("run", p, synth.t(0))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1 and rep.blocked == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT field, reason, record_uri FROM record_refusal").fetchone()
    assert row[0] == "serving_path" and "vllm" in row[1] and row[2].endswith("#1")
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM record_delivery").fetchone()[0] == "refused"
    conn.close()
    assert os.path.exists(os.path.join(root, "state", "refused.jsonl"))
    assert any("serving_path" in r for r in rep.refusals)


def test_a_placeholder_code_sha_run_is_refused_into_record_refusal_and_folds_nothing(tmp_path):
    """End to end: a `run` record whose code_sha is forty `a`s lands in record_refusal naming `code_sha`,
    no run row folds, nothing is blocked, the delivery reads refused; the same run under a derived sha folds."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _write(root, "x", "000001-run.jsonl", [synth.rec("run", synth.run_payload("k", i=0, code_sha="a" * 40), synth.t(0))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1 and rep.blocked == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT field, reason, record_uri FROM record_refusal").fetchone()
    assert row[0] == "code_sha" and "D650" in row[1] and "placeholder" in row[1] and row[2].endswith("#1")
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM record_delivery WHERE file_uri LIKE '%000001-run%'").fetchone()[0] == "refused"
    conn.close()
    assert any("code_sha" in r for r in rep.refusals)
    _write(root, "x", "000002-run.jsonl", [synth.rec("run", synth.run_payload("k", i=0), synth.t(1))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert [r[0] for r in conn.execute("SELECT code_sha FROM run").fetchall()] == [synth.CODE_SHA]
    conn.close()


def test_child_before_parent_is_blocked_then_ingested(tmp_path):
    """Reverse-order delivery: run_state, item_result and judgment arrive before their run."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="judged")
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    children = [r for r in recs if r["record_type"] != "run"]
    items = synth.item_records(rid, jid, 120, 5)
    _write(root, "c", "000001-children.jsonl", children)
    _write(root, "c", "000002-item_result.jsonl.gz", items, gz=True)
    rep = H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    blocked = conn.execute("SELECT record_type, missing_parent_table, missing_parent_id FROM blocked_records").fetchall()
    assert blocked and all(b[1] == "run" and b[2] == rid for b in blocked)
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 0
    conn.close()
    # the parent arrives
    _write(root, "c", "000003-run.jsonl", [r for r in recs if r["record_type"] == "run"])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "ingested"
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM run_state_event").fetchone()[0] == 4
    conn.close()
    assert rep2.unblocked >= len(children) + 5


def test_cross_producer_order(tmp_path):
    """A grading host judgment (written LATER by the clock) arrives BEFORE the cluster's generated record."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="judged")
    cluster = [r for r in recs if r["producer"]["substrate"] == "grading_host" and r["record_type"] in ("run", "run_state", "run_patch")]
    grader = [r for r in recs if r["record_type"] in ("judgment", "run_metric")]
    # deliver the grader's records first, in their own directory (a different producer host)
    _write(root, "grader", "000001-judgment.jsonl", grader)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.blocked == len(grader)
    _write(root, "cluster", "000001-run.jsonl", cluster)
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "ingested"
    assert conn.execute("SELECT COUNT(*) FROM run_metric").fetchone()[0] == 1
    conn.close()


def test_blocked_becomes_stale_after_7_days(tmp_path):
    root = _root(tmp_path)
    st = synth.rec("run_state", {"run_id": "orphan", "state": "running", "event_utc": synth.t(0), "evidence_kind": "job_log"}, synth.t(0))
    _write(root, "o", "000001-run_state.jsonl", [st])
    H.ingest_once(root, workers=0, min_free=0, now="2026-02-01T00:00:00.000Z")
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT stale FROM blocked_records").fetchone()[0] == 0
    conn.close()
    H.ingest_once(root, workers=0, min_free=0, now="2026-02-09T00:00:00.000Z")
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT stale, attempts FROM blocked_records").fetchone()
    assert row[0] == 1 and row[1] >= 2
    conn.close()


def test_run_patch_verify_mismatch_refused_naming_both_values(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="running")
    _write(root, "c", "000001-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    bad = synth.rec("run_patch", {"run_id": rid, "verify": {"serving_path": "lens"}}, synth.t(150))
    _write(root, "c", "000002-patch.jsonl", [bad])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    reason = conn.execute("SELECT field, reason FROM record_refusal").fetchone()
    assert reason[0] == "serving_path" and "'lens'" in reason[1] and "'lora_native'" in reason[1] and rid in reason[1]
    # the matching verify from the lifecycle was recorded in extra_json
    extra = json.loads(conn.execute("SELECT extra_json FROM run").fetchone()[0])
    assert extra["verified_fields"] == {"serving_path": "lora_native"}
    assert conn.execute("SELECT serving_path FROM run").fetchone()[0] == "lora_native"
    conn.close()


def test_run_patch_without_parent_blocked_then_folded(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="queued")
    patch = synth.rec("run_patch", {"run_id": rid, "fields": {"node_hostname": "TESTHOST"}}, synth.t(101))
    _write(root, "c", "000001-patch.jsonl", [patch])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.blocked == 1
    _write(root, "c", "000002-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT node_hostname FROM run").fetchone()[0] == "TESTHOST"
    conn.close()


def test_n_items_planned_patch_folds_with_field_versions_provenance_and_moves_completion(tmp_path):
    """The motivating case: a per-behavior run registered with the WHOLE item set's count (60) where the documented per-run
    count is 10. The patch folds like any patch -- the run reads 10, `field_versions_json.n_items_planned` carries the patch's order
    key -- and, because the completion predicate reads COALESCE(n_items_generated, n_items_planned), a generated run short of its
    planned count becomes complete. A later full `run` record that still carries 60 re-sets it (the fold is by order key), so the
    patch is written after the producer's last `run` record."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="generated", n_items=10, n_items_planned=60)
    _write(root, "c", "000001-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT n_items_planned, n_items_generated FROM run").fetchone()) == (60, 10)
    conn.close()
    patch = synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_planned": 10}}, synth.t(150))
    _write(root, "c", "000002-patch.jsonl", [patch])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT n_items_planned FROM run").fetchone()[0] == 10
    fv = json.loads(conn.execute("SELECT field_versions_json FROM run").fetchone()[0])
    assert fv["n_items_planned"] == [synth.t(150), R.record_sha(patch)]
    conn.close()
    # a re-emitted full run record (newer) still carrying 60 re-sets the count: the fold is by order key, the patch does not pin
    rerun = [r for r in recs if r["record_type"] == "run"][0]
    rerun = dict(rerun, written_utc=synth.t(160))
    _write(root, "c", "000003-run-reemitted.jsonl", [rerun])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT n_items_planned FROM run").fetchone()[0] == 60
    conn.close()


def test_a_top_level_extra_on_a_run_patch_is_not_folded_the_documented_contract(tmp_path):
    """The contract fact, pinned so history keeps folding exactly as it did: `_validate_run_patch`
    reads a patch's `fields` and `verify` only, so a top-level `extra` on a `run_patch` (the writer's central stamps before the routing
    change; 1,634 such records on 668 runs in the log) is DROPPED -- the patch itself folds, its `fields` apply, and the run's `extra_json`
    never sees the keys. The same keys under `fields.extra_json` fold by the additive merge. A rebuild reproduces both."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="running")
    _write(root, "c", "000001-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    stamps = {"run_overrides": {"id": "legacy_stamp_v1", "sha256": "c" * 64, "keys": {"OVERRIDE_ID": "legacy_stamp_v1"}}, "cluster_label": "TESTSITE"}
    legacy = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "legacy"}, "extra": dict(stamps)}, synth.t(150))
    _write(root, "c", "000002-legacy-patch.jsonl", [legacy])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT job_name, extra_json, field_versions_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert row[0] == "legacy"                                            # the patch folded
    extra = json.loads(row[1])
    assert "run_overrides" not in extra and "cluster_label" not in extra      # the top-level extra did not
    assert json.loads(row[2])["job_name"] == [synth.t(150), R.record_sha(legacy)]
    conn.close()
    # the routed form: the same stamps under fields.extra_json reach the run's extra_json (additive merge)
    routed = synth.rec("run_patch", {"run_id": rid, "fields": {"extra_json": dict(stamps)}}, synth.t(160))
    _write(root, "c", "000003-routed-patch.jsonl", [routed])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    extra = json.loads(conn.execute("SELECT extra_json FROM run WHERE run_id = ?", (rid,)).fetchone()[0])
    assert extra["run_overrides"] == stamps["run_overrides"] and extra["cluster_label"] == "TESTSITE"
    assert extra["verified_fields"] == {"serving_path": "lora_native"}        # the lifecycle's own verify, kept beside
    conn.close()
    live = DB.connect(DB.db_path(root), readonly=True)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = H.verify_equal(live, rebuilt)
    assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    sql = "SELECT job_name, extra_json FROM run WHERE run_id = ?"
    assert tuple(live.execute(sql, (rid,)).fetchone()) == tuple(rebuilt.execute(sql, (rid,)).fetchone())
    live.close()
    rebuilt.close()


def test_a_run_patch_whose_fields_extra_json_carries_no_keys_is_refused_by_name_and_never_folds_unstamped(tmp_path):
    """`fields.extra_json` is the ONE place a patch's stamps and provenance ride, and
    `_normalise_payload` keeps only an OBJECT's keys -- so a serialized array (`'[]'`), text that is not JSON, or a bare list beside a
    valid column passed the validator with the value dropped silently: the typed column folded and the writer's `run_overrides` /
    `cluster_label` stamps were lost. The shape is refused BY NAME now (`records.run_patch_extra_json`, the rule the writer shares):
    nothing folds and the refusal names `extra_json`; a serialized object and a dict still park their keys; an explicit null still
    passes. End to end, the record the earlier writer shaped is refused at ingest, the run keeps its row, and the refusal is
    in `record_refusal` by field."""
    for value in ("[]", "[1, 2]", "{not json", '"text"', "7", [], [1], 7, True):
        out = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"job_name": "unstamped", "extra_json": value}}, synth.t(0)))
        assert isinstance(out, VAL.Malformed) and out.field == "extra_json", (value, out)
        assert "object" in out.reason and str(value) not in out.reason, out.reason          # the shape, never the value
    out = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"extra_json": "[]"}}, synth.t(0)))       # alone, too
    assert isinstance(out, VAL.Malformed) and out.field == "extra_json"
    for value, keys in (('{"decision": "D635"}', {"decision": "D635"}), ({"decision": "D635"}, {"decision": "D635"}), ("{}", {}), ({}, {})):
        v = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"job_name": "stamped", "extra_json": value}}, synth.t(0)))
        assert isinstance(v, VAL.Validated), (value, v)
        assert v.payload["fields"] == {"job_name": "stamped"} and v.extra == keys
    v = _v(synth.rec("run_patch", {"run_id": "r", "fields": {"job_name": "null", "extra_json": None}}, synth.t(0)))
    assert isinstance(v, VAL.Validated) and v.extra == {} and v.payload["fields"]["job_name"] == "null"
    # the pure rule both ends share
    assert R.run_patch_extra_json(None) == (None, None)
    assert R.run_patch_extra_json({"a": 1}) == ({"a": 1}, None)
    assert R.run_patch_extra_json('{"a": 1}') == ({"a": 1}, None)
    for value in ("[]", "{not json", [], 7):
        obj, reason = R.run_patch_extra_json(value)
        assert obj is None and reason and "run_patch fields.extra_json" in reason, (value, reason)
    # end to end: a valid column beside a serialized array, as the pre-review writer would have filed it with the stamps lost
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="running")
    _write(root, "c", "000001-run.jsonl", recs)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    before = conn.execute("SELECT job_name, extra_json, field_versions_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    conn.close()
    malformed = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "unstamped", "extra_json": "[]"}}, synth.t(150))
    _write(root, "c", "000002-malformed-patch.jsonl", [malformed])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1 and rep.blocked == 0, (rep.refused, rep.blocked, rep.refusals)
    assert any("extra_json" in r for r in rep.refusals), rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT job_name, extra_json, field_versions_json FROM run WHERE run_id = ?", (rid,)).fetchone()) == tuple(before)
    ref = conn.execute("SELECT record_type, field, reason FROM record_refusal WHERE record_sha256 = ?", (R.record_sha(malformed),)).fetchone()
    assert ref is not None and ref[0] == "run_patch" and ref[1] == "extra_json" and "object" in ref[2], ref
    conn.close()
    # the same column under a well-formed extra_json folds, provenance and all
    good = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "stamped", "extra_json": {"run_overrides": {"id": "v1"}}}}, synth.t(160))
    _write(root, "c", "000003-good-patch.jsonl", [good])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT job_name, extra_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert row[0] == "stamped" and json.loads(row[1])["run_overrides"] == {"id": "v1"}
    conn.close()


def test_materialized_row_validated_after_patch(tmp_path):
    """A patch whose folded result violates a CHECK (gpu_count negative? no — use walltime_source without walltime) is refused."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="running")
    _write(root, "c", "000001-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    # a patch carrying an out-of-vocabulary value is refused by the validator; a patch that passes the
    # validator but breaks the row (a served_max_num_seqs_source with no vocabulary match) is refused by the CHECK
    bad = synth.rec("run_patch", {"run_id": rid, "fields": {"served_max_num_seqs_source": "guessed"}}, synth.t(150))
    _write(root, "c", "000002-patch.jsonl", [bad])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert "served_max_num_seqs_source" in conn.execute("SELECT reason FROM record_refusal").fetchone()[0]
    assert conn.execute("SELECT served_max_num_seqs_source FROM run").fetchone()[0] is None
    conn.close()


def test_record_file_reading_plain_and_gz_and_logical_sha(tmp_path):
    recs = synth.seed_dimensions()[:3]
    plain = R.write_records(str(tmp_path / "a.jsonl"), recs)
    gz = R.write_records(str(tmp_path / "b.jsonl"), recs, gz=True)
    assert gz.endswith(".gz")
    assert R.file_logical_sha(plain) == R.file_logical_sha(gz)
    assert [s for _, _, s in R.read_record_file(plain)] == [s for _, _, s in R.read_record_file(gz)]
    assert R.logical_uri("x/y.jsonl.gz") == "x/y.jsonl"
    with pytest.raises(ValueError):
        (tmp_path / "bad.jsonl").write_text("{not json\n")
        list(R.read_record_file(str(tmp_path / "bad.jsonl")))
