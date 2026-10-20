"""The relayed rulings (2026-09-17, the authors' paper-inclusion criteria, relayed): the committed seed carries them
with `provenance.kind = relayed` and the loader ingests them; a synthetic batch-8 original reads `excluded` in `v_paper_current`
with the decisive ruling id, and the ruling row says who it came from; a synthetic distilled ReLU difference reads `excluded`
with its landing condition unexpired (`not_evaluated` before the harvester's pass, `missing` after it: no comparison record
yet, the exclusion stays active); a synthetic seed replicate named by the include reads `selected` although no roster entry
matches it; a ruling ruled after the provenance cutoff with no provenance block, or with an unknown kind, REFUSES by name; the status
page lists the relayed rulings first with their message; `ruling add` stamps the block.
"""
from __future__ import annotations

import json
import os

import pytest
import yaml

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry.adapters import roster as RO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROSTER = os.path.join(ROOT, "configs", "registry", "roster_r2_v1.yaml")
RULINGS = os.path.join(ROOT, "configs", "registry", "rulings_r2_v1.jsonl")

PAPER_INCLUSION_IDS = {
    "r2v1-exclude-qwen-batch8-base-checkpoint-originals": "exclude",
    "r2v1-exclude-qwen-distilled-relu-difference-pending-relu-comparison": "exclude",
    "r2v1-include-qwen-batch64-seed-replicates-fitted": "include",
    "r2v1-include-qwen-batch64-seed-replicates-derived": "include",
    "r2v1-operating_point-qwen-arditi_allpos-c0p3-user-override": "operating_point",
    "r2v1-exclude-qwen-arditi_allpos-other-rungs": "exclude",
    "r2v1-include-qwen-bon-vendor_preset-new-instrument": "include",
    "decode-qwen-vendor-preset-bon-multiturn-petri-2026-09-17": "reference",
}
POST_PIN_IDS = {
    "r2v1-exclude-not_run_by_design-qwen-alphasteer_rebake-agentic_leaking": "peer_encoded",
    "r2v1-include-qwen-b2i-leaking-cross-engine-admission-native-relu": "peer_encoded",
    "r2v1-include-qwen-arditi_allpos-ar_c0p3-served-operating-point": "user_direct",
}
#: the two 2026-09-17 decode rulings (registry commit 294b774e): ruled directly, so their blocks are `user_direct` and confirmed;
#: date-precision stamps (the rulings are the day's, not a message's)
DECODE_PINS_IDS = {
    "decode-bon-vendor-preset-2026-09-17": "user_direct",
    "decode-nemotron-vendor-preset-0.95-2026-09-17": "user_direct",
}
#: the standing decode rule of 2026-09-17 (use the vendor presets unless instructed otherwise), relayed at 2026-09-18T02:24Z with the
#: finding that motivates it (em8q had silently kept T = 1.0); filed with a `relayed` block, its reading not yet confirmed
EM_RELAYED_IDS = {
    "decode-vendor-preset-default-2026-09-17": "reference",
}
#: the Llama thinking-equivalence ruling (2026-09-18, ruled directly): `user_direct`
THINKING_NA_IDS = {"decode-llama-thinking-not-available-2026-09-18": "user_direct"}
WITHHELD = "not published"
KEPT_PROVENANCE_KEYS = {"kind", "message_utc", "commit", "commit_utc", "confirmed_by_user", "relayed_by", "encoded_by"}
BATCH8_LORA = "63c151654c3d9cbdd9540b1806e7db81a6e2434f509ba6657906beccf8762ffd"     # base__constv2-mix-full__lora_r4_ce on qwen35_4b_base
REPLICATE_S1 = "8be1e1a7ab36f3f65557f7835a30c6b8f18143c781fbf7fb312a5cc7a6e08871"   # base64s1__constv2-mix-full__lora_r4_ce


# --------------------------------------------------------------------------------------------
# the committed seed
# --------------------------------------------------------------------------------------------


def test_committed_seed_carries_the_relayed_rulings_with_provenance():
    load = RO.load(ROSTER, RULINGS)
    by_id = {r["ruling_id"]: r for r in load.rulings}
    for rid, action in PAPER_INCLUSION_IDS.items():
        r = by_id[rid]
        assert r["action"] == action, rid
        pv = r["extra"]["provenance"]
        assert pv["kind"] == "relayed" and pv["relayed_by"] == WITHHELD and pv["confirmed_by_user"] is False, rid
        assert pv["message_utc"].startswith("2026-09-17T18:") and r["ruled_utc"] >= RO.PROVENANCE_REQUIRED_FROM_UTC, rid
        assert r["ruled_by"] == "the authors" and r["source_ref"] == WITHHELD and set(pv) <= KEPT_PROVENANCE_KEYS, rid
    # the three post-pin (f33dca5d) rulings: two encoded by a commit, one ruled directly through the selection record
    for rid, kind in POST_PIN_IDS.items():
        pv = by_id[rid]["extra"]["provenance"]
        assert pv["kind"] == kind and pv["confirmed_by_user"] is False, rid
        if kind == "peer_encoded":
            assert pv["encoded_by"] == WITHHELD and pv["commit"] == "599b3bed"
        assert set(pv) <= KEPT_PROVENANCE_KEYS, rid
    nrbd = by_id["r2v1-exclude-not_run_by_design-qwen-alphasteer_rebake-agentic_leaking"]
    assert nrbd["scope_json"]["match"] == {"model_ids": ["qwen35_4b_it"], "paper_labels": ["alphasteer_rebake"], "eval_ids": ["agentic_leaking"]}
    assert nrbd["extra"]["disposition_table_state"] == "not_applicable" and nrbd["ruled_utc"] == "2026-09-16T00:00:00.000Z"
    xe = by_id["r2v1-include-qwen-b2i-leaking-cross-engine-admission-native-relu"]
    assert xe["scope_json"]["arms_absent_by_ruling"] == ["alphasteer"] and xe["scope_json"]["unrostered"] == ["none__lens0"]
    assert xe["scope_json"]["match"]["transfer_kinds"] == ["base_to_instruct"] and len(xe["scope_json"]["native_arms_legacy"]) == 3
    assert set(xe["extra"]["provenance"]) <= KEPT_PROVENANCE_KEYS
    ap = by_id["r2v1-include-qwen-arditi_allpos-ar_c0p3-served-operating-point"]
    assert ap["scope_json"]["match"] == {"model_ids": ["qwen35_4b_it"], "object_ids": ["ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"]}
    assert ap["target_json"]["served"]["c"] == 0.3 and ap["target_json"]["rule"]["view"] == "ar_c0p2"
    # (d) the fig5 source-preference ruling's scope text names the post-pin MMLU source; its date and author are the original's
    f5 = by_id["r2v1-include-fig5-source-preference-per-lane"]
    assert "strength_lane_base2it_native_mmlu" in f5["target_json"]["defaults_by_component_dose"]["(capability, 0.5)"]
    assert "strength_lane_base2it_native_mmlu" in f5["target_json"]["overrides"]["gemma4e4b_it / qwen35_4b_it / llama8b_it base_to_instruct"]
    assert f5["ruled_utc"] == "2026-09-16T00:00:00.000Z" and "provenance" not in f5["extra"] and f5["extra"]["scope_text_refreshed_at_pin"].startswith("f33dca5d")
    # the two decode rulings ruled directly on 2026-09-17: reference rulings that bind decode_policy rows (BoN at the
    # vendor preset; Nemotron's preset = the shipped 0.95), user_direct and confirmed, dated (not timed), covering no run
    for rid, kind in DECODE_PINS_IDS.items():
        r = by_id[rid]
        pv = r["extra"]["provenance"]
        assert pv["kind"] == kind and pv["confirmed_by_user"] is True and set(pv) <= KEPT_PROVENANCE_KEYS, rid
        assert r["action"] == "reference" and r["scope_kind"] == "key" and r["scope_json"]["match"] is None, rid
        assert r["target_json"]["decode_policy_ruling"] is True and r["expires_when_json"] is None, rid
        assert r["ruled_utc"] == "2026-09-17T00:00:00.000Z" and r["ruled_utc_precision"] == "date" and r["ruled_by"] == "the authors", rid
        assert r["source_ref"] == WITHHELD, rid
    assert "instruments were counted at 0.95" in by_id["decode-nemotron-vendor-preset-0.95-2026-09-17"]["reason"]
    assert "not being re-run" in by_id["decode-nemotron-vendor-preset-0.95-2026-09-17"]["reason"]
    # the relayed standing ruling (2026-09-17 19:24 PDT): a reference ruling binding the em8q vendor_preset default on qwen35_4b_it and
    # every future default (the global test tests/test_registry_decode_defaults_vendor.py enumerates the instructed exceptions); relayed,
    # unconfirmed, timed at the relay message
    for rid, action in EM_RELAYED_IDS.items():
        r = by_id[rid]
        pv = r["extra"]["provenance"]
        assert r["action"] == action and r["scope_kind"] == "key" and r["scope_json"]["match"] is None and r["expires_when_json"] is None, rid
        assert pv["kind"] == "relayed" and pv["relayed_by"] == WITHHELD and pv["message_utc"] == "2026-09-18T02:24Z" and pv["confirmed_by_user"] is False, rid
        assert set(pv) <= KEPT_PROVENANCE_KEYS and r["ruled_by"] == "the authors", rid
        assert r["ruled_utc"] == "2026-09-18T02:24:00.000Z" and r["ruled_utc_precision"] == "datetime", rid
        assert r["target_json"]["decode_policy_ruling"] is True and "qwen35_4b_it|em8q|off|vendor_preset@v1" in r["target_json"]["rows"], rid
        assert "57.8" in r["target_json"]["finding"] and r["source_ref"] == WITHHELD, rid
    # the header marks the post-pin section without re-pinning the file; the decode-rulings batch beside it, in the same shape
    header0, _ = RO.read_rulings_jsonl(RULINGS)
    assert header0["paper_build_commit"].startswith("40997227") and set(header0["post_pin_additions"]["f33dca5d"]["added"]) == set(POST_PIN_IDS)
    assert set(header0["post_pin_additions"]["294b774e"]["added"]) == set(DECODE_PINS_IDS)
    assert set(header0["post_pin_additions"]["registry-em-organisms"]["added"]) == set(EM_RELAYED_IDS)
    assert set(header0["post_pin_additions"]["registry-llama-thinking-na"]["added"]) == set(THINKING_NA_IDS)
    # the AlphaSteer x leaking cells on qwen are dispositions (not run by design), never gaps: four entries, both lanes
    disp = [d for d in load.dispositions if d["model_id"] == "qwen35_4b_it" and d["eval_id"] == "agentic_leaking" and (d["extra"] or {}).get("paper_label") == "alphasteer_rebake"]
    assert sorted(d["comparison_id"] for d in disp) == ["fig3_misalign/base_to_instruct", "fig3_misalign/instruct_direct",
                                                        "figA_misalignment_absolute/base_to_instruct", "figA_misalignment_absolute/instruct_direct"]
    assert all(d["state"] == "not_applicable" and "NOT_RUN_BY_DESIGN" in d["reason"] and "f33dca5d" in d["reason"] for d in disp)
    # none of the earlier rulings was re-stamped
    assert all("provenance" not in (r["extra"] or {}) for r in load.rulings
               if r["ruling_id"] not in PAPER_INCLUSION_IDS and r["ruling_id"] not in POST_PIN_IDS and r["ruling_id"] not in DECODE_PINS_IDS
               and r["ruling_id"] not in EM_RELAYED_IDS and r["ruling_id"] not in THINKING_NA_IDS)
    for rid, kind in THINKING_NA_IDS.items():
        r = next(x for x in load.rulings if x["ruling_id"] == rid)
        assert r["extra"]["provenance"]["kind"] == kind and r["extra"]["provenance"]["confirmed_by_user"] is True
        assert set(r["extra"]["provenance"]) <= KEPT_PROVENANCE_KEYS and r["action"] == "reference"
    # (a) names the batch-8 originals by id; (c) the twelve fitted replicates by id, the derived ones by selector
    a = by_id["r2v1-exclude-qwen-batch8-base-checkpoint-originals"]
    assert BATCH8_LORA in a["scope_json"]["match"]["object_ids"] and a["scope_json"]["match"]["model_ids"] == ["qwen35_4b_it"]
    assert a["target_json"]["never_rerun"] is True
    c1 = by_id["r2v1-include-qwen-batch64-seed-replicates-fitted"]
    assert len(c1["scope_json"]["match"]["object_ids"]) == 12 and REPLICATE_S1 in c1["scope_json"]["match"]["object_ids"]
    c2 = by_id["r2v1-include-qwen-batch64-seed-replicates-derived"]
    assert c2["scope_json"]["match"] == {"model_ids": ["qwen35_4b_it"], "transfer_kinds": ["base_to_instruct"], "object_kinds": ["lora_subtraction"], "campaigns": ["reseed"]}
    # (b) is self-retiring on the comparison record, and says why it is an exclude
    b = by_id["r2v1-exclude-qwen-distilled-relu-difference-pending-relu-comparison"]
    ew = b["expires_when_json"]
    assert ew["kind"] == "any_of" and len(ew["conditions"]) == 1 and ew["conditions"][0]["kind"] == "served_id_absent" and ew["guards"] == []
    assert "psr_ce_all_minus_ctl_paired" in ew["conditions"][0]["input_uri"] and "hold_out" in b["target_json"]["why_exclude_not_hold_out"]
    # (d) binds the c = 0.3 point and its selection outcome; (e, reversed) names the new decode row and the separate comparison
    d = by_id["r2v1-operating_point-qwen-arditi_allpos-c0p3-user-override"]
    assert d["target_json"]["strength_coefficient"] == 0.3 and d["target_json"]["selection_outcome"] == "user_override_over_budget_2026-09-16"
    e = by_id["r2v1-include-qwen-bon-vendor_preset-new-instrument"]
    assert e["target_json"]["decode_policy_id"] == "qwen35_4b_it|bon|off|vendor_preset@v1" and e["target_json"]["separate_comparison"] is True
    assert e["extra"]["reversal_of"]["reversed_at"] == "2026-09-17T18:58:00Z"
    # the header's counts agree with the file, and show_text lists the relayed ones
    header, raw = RO.read_rulings_jsonl(RULINGS)
    # 89 + the two 2026-09-17 decode rulings + the relayed vendor-preset default ruling
    # + the 2026-09-18 Llama thinking-equivalence ruling (decode-llama-thinking-not-available-2026-09-18, user_direct)
    assert header["counts_by_action"]["total"] == len(raw) == 93 and set(header["added_after_the_draft"]["paper_inclusion_criteria"]) == set(PAPER_INCLUSION_IDS)
    text = RO.show_text(load)
    assert "rulings by provenance" in text and sum(1 for line in text.splitlines() if line.startswith("  relayed  ")) == 8 + len(EM_RELAYED_IDS)
    # the roster's addendum names the rulings, and the qwen arditi entries carry the c = 0.3 annotation
    ry = yaml.safe_load(open(ROSTER))
    assert set(ry["meta"]["addendum_2026_09_17"]["rulings_added"]) == set(PAPER_INCLUSION_IDS)
    pin = ry["objects"]["pins"]["arditi_allpos_operating"]["qwen35_4b_it"]
    assert pin["coefficient"] == 0.3 and pin["operating_point_id_legacy"] == "ar_c0p3__arditi_allpos"
    qwen_arditi = [x for x in load.entries if x["model_id"] == "qwen35_4b_it" and (x["object_selector_json"] or {}).get("paper_label") == "arditi"]
    assert qwen_arditi and all(x["inferred"] == 1 and x["extra"]["object_pin"] == "arditi_allpos_operating" for x in qwen_arditi)
    assert all(x["extra"]["object_pin_detail"]["object_id_prefix"] == "ba2212f0" for x in qwen_arditi)


# --------------------------------------------------------------------------------------------
# the loader's provenance rule
# --------------------------------------------------------------------------------------------


def _ruling(**over):
    d = {"ruling_id": "R_x", "round_id": "r_synth", "action": "exclude", "scope_kind": "key", "scope_json": {"model_id": synth.MODEL_ID},
         "target_json": {}, "reason": "synthetic", "ruled_utc": "2026-09-18T10:00:00Z", "ruled_by": "the user", "source_ref": "test"}
    d.update(over)
    return d


def test_a_ruling_after_the_cutoff_without_provenance_refuses_by_name():
    with pytest.raises(RO.RosterRefusal, match="provenance missing"):
        RO.normalise_ruling(_ruling())
    with pytest.raises(RO.RosterRefusal, match="provenance kind 'hearsay'"):
        RO.normalise_ruling(_ruling(provenance={"kind": "hearsay"}))
    with pytest.raises(RO.RosterRefusal, match="needs relayed_by"):
        RO.normalise_ruling(_ruling(provenance={"kind": "relayed", "message_utc": "2026-09-18T09:00:00Z"}))
    with pytest.raises(RO.RosterRefusal, match="needs message_utc"):
        RO.normalise_ruling(_ruling(provenance={"kind": "relayed", "relayed_by": "peer"}))
    with pytest.raises(RO.RosterRefusal, match="neither a date nor an ISO-8601"):
        RO.normalise_ruling(_ruling(provenance={"kind": "relayed", "relayed_by": "peer", "message_utc": "yesterday"}))
    with pytest.raises(RO.RosterRefusal, match="needs encoded_by"):
        RO.normalise_ruling(_ruling(provenance={"kind": "peer_encoded", "commit": "599b3bed"}))
    with pytest.raises(RO.RosterRefusal, match="is not a git sha"):
        RO.normalise_ruling(_ruling(provenance={"kind": "peer_encoded", "encoded_by": "peer", "commit": "yesterday's build"}))
    assert RO.normalise_ruling(_ruling(provenance={"kind": "peer_encoded", "encoded_by": "peer", "commit": "599b3bed"}))["extra"]["provenance"] == \
        {"kind": "peer_encoded", "encoded_by": "peer", "commit": "599b3bed", "confirmed_by_user": False}
    # a well-formed block is parked in extra.provenance with the confirmation flag defaulted
    p = RO.normalise_ruling(_ruling(provenance={"kind": "relayed", "relayed_by": "peer", "message_utc": "2026-09-18T09:00:00Z"}))
    assert p["extra"]["provenance"] == {"kind": "relayed", "relayed_by": "peer", "message_utc": "2026-09-18T09:00:00Z", "confirmed_by_user": False}
    assert RO.normalise_ruling(_ruling(provenance={"kind": "user_direct"}))["extra"]["provenance"] == {"kind": "user_direct"}
    # before the cutoff the block is optional (the transcribed seed) but still validated when present
    assert "provenance" not in RO.normalise_ruling(_ruling(ruled_utc="2026-09-11"))["extra"]
    with pytest.raises(RO.RosterRefusal, match="provenance kind 'rumour'"):
        RO.normalise_ruling(_ruling(ruled_utc="2026-09-11", provenance={"kind": "rumour"}))


# --------------------------------------------------------------------------------------------
# the paper flag on synthetic runs covered by the relayed rulings
# --------------------------------------------------------------------------------------------


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _roster_entry(i, object_selector, comparison="fig_synth"):
    return synth.rec("roster_entry", {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID,
                                      "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID,
                                      "object_selector_json": object_selector, "strength_rule": "operating_point",
                                      "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}, synth.t(50 + i))


def _object(sha, kind, label, role, campaign="sdfbase_full22", replicate=None):
    o = synth.object_lora(sha=sha, paper_label=label, role=role)
    o.update(object_kind=kind, campaign=campaign)
    if replicate:
        o["fit_replicate_label"] = replicate
    return o


def _relayed(rid, action, scope_kind, scope, i, ew=None, **extra):
    raw = {"ruling_id": rid, "round_id": "r_synth", "action": action, "scope_kind": scope_kind, "scope_json": scope, "target_json": {},
           "reason": "the user's criterion, relayed", "ruled_utc": "2026-09-17T18:48:00Z", "ruled_by": "the user, relayed by peer",
           "source_ref": "test", "expires_when_json": ew,
           "provenance": {"kind": "relayed", "relayed_by": "peer", "message_utc": "2026-09-17T18:48:00Z", "message_local": "11:48 PDT"}}
    raw.update(extra)
    return synth.rec("ruling", RO.normalise_ruling(raw), synth.t(i))


@pytest.fixture
def world(tmp_path):
    """Three fitted objects on the synthetic model: a batch-8 original, a distilled ReLU difference and a seed-1 replicate, each with
    one judged run; one roster entry matches the batch-8 and the distilled objects by label; the replicate matches NO entry."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    b8 = "b8" * 32
    dist = "d1" * 32
    rep = "51" * 32
    dims = [synth.rec("object", _object(b8, "lora", "constitution_lora", "constitution"), synth.t(12)),
            synth.rec("object", _object(dist, "relu_steer_distilled_difference", "difference_relu", "difference"), synth.t(13)),
            synth.rec("object", _object(rep, "lora", "other", "constitution", campaign="reseed", replicate="s1"), synth.t(14))]
    r_b8, recs_b8 = synth.run_records("store_b8", 900, "constitution_lora@b8", b8, 3)
    r_dist, recs_dist = synth.run_records("store_dist", 1000, "difference_relu@d1", dist, 3)
    r_rep, recs_rep = synth.run_records("store_rep", 1100, "replicate@51", rep, 3)
    paired = "synthetic:///relu_paired/synthmodel__syntheval__constv2-mix-full__psr_ce_all_minus_ctl_paired.json"
    rulings = [
        _relayed("R_a_batch8_out", "exclude", "object", {"model_id": synth.MODEL_ID, "match": {"model_ids": [synth.MODEL_ID], "object_ids": [b8]}}, 60),
        _relayed("R_b_distilled_out", "exclude", "object",
                 {"model_id": synth.MODEL_ID, "match": {"model_ids": [synth.MODEL_ID], "object_kinds": ["relu_steer_distilled_difference"]}}, 61,
                 ew={"kind": "any_of", "conditions": [{"kind": "served_id_absent", "input_uri": paired, "member_key": "$.status", "served_id": "pending"}], "guards": []}),
        _relayed("R_c_replicates_in", "include", "object", {"model_id": synth.MODEL_ID, "match": {"model_ids": [synth.MODEL_ID], "object_ids": [rep]}}, 62),
    ]
    entries = [_roster_entry(1, {"paper_label": "constitution_lora"}), _roster_entry(2, {"paper_label": "difference_relu"}, comparison="fig_q")]
    _write(root, "d83", "000001-d83.jsonl", dims + recs_b8 + recs_dist + recs_rep + entries + rulings)
    rep_ = H.ingest_once(root, workers=0, min_free=0)
    assert rep_.refused == 0, rep_.refusals
    return root, {"b8": r_b8, "dist": r_dist, "rep": r_rep, "paired": paired}


def _pc(conn, run_id):
    return dict(conn.execute("SELECT * FROM v_paper_current WHERE run_id = ?", (run_id,)).fetchone())


def test_batch8_original_is_excluded_by_the_relayed_ruling_and_the_row_says_who_ruled(world):
    root, ids = world
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn, ids["b8"])
    assert pc["selection_state"] == "excluded" and pc["in_current_paper"] == 0 and pc["decisive_ruling_id"] == "R_a_batch8_out"
    row = dict(conn.execute("SELECT evidence_state, expired_utc, extra_json FROM ruling WHERE ruling_id = 'R_a_batch8_out'").fetchone())
    pv = json.loads(row["extra_json"])["provenance"]
    assert pv["kind"] == "relayed" and pv["relayed_by"] == "peer" and pv["confirmed_by_user"] is False
    # the DDL's evidence_state is the EXPIRY evidence, untouched on a ruling with no expiry: it never says `relayed`
    assert row == dict(row, evidence_state="not_evaluated", expired_utc=None)
    conn.close()


def test_distilled_difference_is_excluded_with_the_landing_condition_unexpired(world):
    root, ids = world
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn, ids["dist"])
    assert pc["selection_state"] == "excluded" and pc["in_current_paper"] == 0 and pc["decisive_ruling_id"] == "R_b_distilled_out"
    assert pc["held_out"] == 0    # an exclude, not a hold: the ruling says OUT
    row = dict(conn.execute("SELECT evidence_state, expired_utc FROM ruling WHERE ruling_id = 'R_b_distilled_out'").fetchone())
    # no comparison record exists: the harvester's evaluation reads `missing` and the exclusion STAYS ACTIVE
    assert row == {"evidence_state": "missing", "expired_utc": None}
    conn.close()
    # the comparison lands (an adapter files the record's status): the exclusion self-retires and the run is the roster's again
    state = synth.rec("source_state", {"source_id": "relu_native:synth", "input_uri": ids["paired"], "json_path": "$.status", "found": 1,
                                       "path_resolves": 1, "current_members_json": ["ok"], "observed_utc": synth.t(80), "input_sha256": "2" * 64}, synth.t(80))
    _write(root, "d83", "000002-landed.jsonl", [state])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = dict(conn.execute("SELECT evidence_state, expired_utc FROM ruling WHERE ruling_id = 'R_b_distilled_out'").fetchone())
    assert row["evidence_state"] == "ok" and row["expired_utc"] is not None
    assert _pc(conn, ids["dist"])["selection_state"] == "selected"
    conn.close()


def test_seed_replicate_named_by_the_include_is_selected_without_a_roster_match(world):
    root, ids = world
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn, ids["rep"])
    assert pc["selection_state"] == "selected" and pc["in_current_paper"] == 1 and pc["decisive_ruling_id"] == "R_c_replicates_in"
    assert json.loads(pc["comparison_ids_json"]) == []      # no entry matched (paper_label `other`): the include carries it
    # the status page lists the relayed rulings first, with the relaying session and the message time, and counts them on a tile
    ctx = status.status_json(root, conn)
    rows = ctx["rulings"]
    assert [r["ruling_id"] for r in rows][:3] == sorted(["R_a_batch8_out", "R_b_distilled_out", "R_c_replicates_in"], reverse=True) or \
        {r["ruling_id"] for r in rows[:3]} == {"R_a_batch8_out", "R_b_distilled_out", "R_c_replicates_in"}
    assert all(r["provenance"] == "relayed" and r["relayed_by"] == "peer" and r["message_utc"].startswith("2026-09-17T18:48") for r in rows[:3])
    assert len(ctx["rulings_relayed_unconfirmed"]) == 3
    decided = {r["ruling_id"]: r["n_runs_decided"] for r in rows}
    assert decided["R_a_batch8_out"] == 1 and decided["R_c_replicates_in"] == 1
    html = status.render_html(ctx)
    assert "Rulings of the open round" in html and "relayed rulings awaiting the user's confirmation" in html
    conn.close()


def test_ruling_add_cli_stamps_provenance(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    # after the cutoff the CLI's default is user_direct; a relayed one names its message
    rc = cli.main(["ruling", "add", "--round", "r_synth", "--action", "exclude", "--scope-kind", "run", "--scope-json", json.dumps({"run_id": "x"}),
                   "--reason", "test", "--source-ref", "test", "--id", "R_cli_relayed", "--ruled-utc", "2026-09-18T01:02:03Z",
                   "--provenance-kind", "relayed", "--relayed-by", "peer", "--message-utc", "2026-09-18T01:00:00Z"])
    assert rc == 0
    path = capsys.readouterr().out.strip()
    rec = next(iter(R.read_record_file(path)))[1]
    assert rec["payload"]["extra"]["provenance"] == {"kind": "relayed", "relayed_by": "peer", "message_utc": "2026-09-18T01:00:00Z", "confirmed_by_user": False}
    assert cli.main(["ruling", "add", "--round", "r_synth", "--action", "exclude", "--scope-kind", "run", "--scope-json", json.dumps({"run_id": "y"}),
                     "--reason", "test", "--source-ref", "test", "--id", "R_cli_direct", "--ruled-utc", "2026-09-18T01:02:03Z"]) == 0
    path2 = capsys.readouterr().out.strip()
    assert next(iter(R.read_record_file(path2)))[1]["payload"]["extra"]["provenance"] == {"kind": "user_direct"}
    # relayed without the message is refused by name
    assert cli.main(["ruling", "add", "--round", "r_synth", "--action", "exclude", "--scope-kind", "run", "--scope-json", json.dumps({"run_id": "z"}),
                     "--reason", "test", "--source-ref", "test", "--id", "R_cli_bad", "--provenance-kind", "relayed"]) == 1
    assert "needs relayed_by" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# the committed files in a database: the not-run-by-design cells are dispositions, not gaps
# --------------------------------------------------------------------------------------------

QWEN_ALPHASTEER_PIN = "cd9c311bcf4ee184de829bfdad1082d6e5924d2a11c805cb69203bc0614b30f0"   # objects.pins.alphasteer_rebake_operating.qwen35_4b_it


def test_alphasteer_x_leaking_on_qwen_is_a_disposition_not_a_gap_in_v_roster_gaps(tmp_path):
    """The post-pin not-run-by-design ruling's roster exception: with the dimension seeds loaded and the pinned qwen AlphaSteer
    object present, the four qwen35_4b_it x alphasteer_rebake x agentic_leaking entries (fig3 + figA, both lanes) carry a
    `not_applicable` disposition naming NOT_RUN_BY_DESIGN @ f33dca5d and are absent from v_roster_gaps, while the sibling
    agentic_murder entries of the same object (no ruling) are gaps."""
    from sisteer.registry.adapters import dimensions as D
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    D.write_seed_records(root, os.path.join(ROOT, "configs", "registry"), producer=synth.producer("tests.relayed"), now=synth.t(1))
    obj = synth.object_lora(sha=QWEN_ALPHASTEER_PIN, paper_label="alphasteer_rebake", role="baseline_published")
    for k in ("fit_corpus_id", "fit_corpus_sha256", "fit_rank", "fit_alpha"):
        obj.pop(k, None)
    obj.update(object_kind="alphasteer", fit_objective="alphasteer", campaign="alphasteer_f22", fit_target_kind="contrastive_direction",
               fit_model_id="qwen35_4b_it", model_id="qwen35_4b_it",
               fit_contrast_sets_json={"positive": {"id": "harmful_pool", "sha256": synth.SHA256_A, "n": 64}, "negative": {"id": "benign_pool", "sha256": synth.SHA256_B, "n": 64}},
               fit_positions="last_token_of_appended_assistant_header")
    _write(root, "pins", "000001-object.jsonl", [synth.rec("object", obj, synth.t(12))])
    RO.write_records(root, RO.load(ROSTER, RULINGS), written_utc=synth.t(50))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    entries = {r[0]: r[1] for r in conn.execute("SELECT roster_entry_id, comparison_id FROM roster_entry WHERE model_id = 'qwen35_4b_it' AND object_id = ?", (QWEN_ALPHASTEER_PIN,))}
    leaking = conn.execute("SELECT roster_entry_id, comparison_id FROM roster_entry WHERE model_id = 'qwen35_4b_it' AND object_id = ? AND eval_id = 'agentic_leaking'",
                           (QWEN_ALPHASTEER_PIN,)).fetchall()
    assert sorted(r[1] for r in leaking) == ["fig3_misalign/base_to_instruct", "fig3_misalign/instruct_direct",
                                             "figA_misalignment_absolute/base_to_instruct", "figA_misalignment_absolute/instruct_direct"]
    ids = [r[0] for r in leaking]
    marks = ",".join("?" * len(ids))
    disp = conn.execute(f"SELECT roster_entry_id, state, reason FROM disposition WHERE roster_entry_id IN ({marks})", ids).fetchall()
    assert len(disp) == 4 and all(d[1] == "not_applicable" and "NOT_RUN_BY_DESIGN" in d[2] and "f33dca5d" in d[2] for d in disp)
    assert conn.execute(f"SELECT COUNT(*) FROM v_roster_gaps WHERE roster_entry_id IN ({marks})", ids).fetchone()[0] == 0
    # the same object's murder cells have no ruling: gaps (no run in this database)
    murder = [r[0] for r in conn.execute("SELECT roster_entry_id FROM roster_entry WHERE model_id = 'qwen35_4b_it' AND object_id = ? AND eval_id = 'agentic_murder'", (QWEN_ALPHASTEER_PIN,))]
    assert murder and conn.execute(f"SELECT COUNT(*) FROM v_roster_gaps WHERE roster_entry_id IN ({','.join('?' * len(murder))})", murder).fetchone()[0] == len(murder)
    # the ruling itself is in the round with its provenance, and nothing about it is an expiry evaluation
    row = dict(conn.execute("SELECT evidence_state, expired_utc, extra_json FROM ruling WHERE ruling_id = 'r2v1-exclude-not_run_by_design-qwen-alphasteer_rebake-agentic_leaking'").fetchone())
    assert row["evidence_state"] == "not_evaluated" and row["expired_utc"] is None
    assert json.loads(row["extra_json"])["provenance"] == {"kind": "peer_encoded", "encoded_by": WITHHELD, "commit": "599b3bed",
                                                            "commit_utc": "2026-09-17T06:39:22Z", "confirmed_by_user": False}
    conn.close()


def test_distilled_label_matches_no_paper_roster_entry_while_the_paired_write_does(tmp_path):
    """The 2026-09-19 ruling (all five families): the residual-distilled student carries `difference_relu_distilled`, a label no roster
    entry selects, so its judged run is `not_in_roster` / out of the current paper WITHOUT any ruling; the UNDISTILLED paired write (object_kind
    `composed`) carries `difference_relu` and IS selected by the arm's roster entry. The label mapping alone keeps the student out."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=2, n_items=3)
    dist, pw = "d2" * 32, "5e" * 32
    dims = [synth.rec("object", _object(dist, "relu_steer_distilled_difference", "difference_relu_distilled", "difference"), synth.t(12)),
            synth.rec("object", _object(pw, "composed", "difference_relu", "difference"), synth.t(13))]
    r_dist, recs_dist = synth.run_records("store_dist2", 1200, "difference_relu_distilled@d2", dist, 3)
    r_pw, recs_pw = synth.run_records("store_pw", 1300, "difference_relu@5e", pw, 3)
    entries = [_roster_entry(7, {"paper_label": "difference_relu", "object_kind": "composed"}, comparison="fig_q")]
    _write(root, "d19", "000001-d19.jsonl", dims + recs_dist + recs_pw + entries)
    rep_ = H.ingest_once(root, workers=0, min_free=0)
    assert rep_.refused == 0, rep_.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc_d, pc_p = _pc(conn, r_dist), _pc(conn, r_pw)
    assert pc_d["selection_state"] == "not_in_roster" and pc_d["in_current_paper"] == 0 and pc_d["decisive_ruling_id"] is None
    assert pc_p["selection_state"] == "selected" and pc_p["in_current_paper"] == 1
    labels = dict(conn.execute("SELECT object_id, paper_label FROM object WHERE object_id IN (?, ?)", (dist, pw)).fetchall())
    assert labels == {dist: "difference_relu_distilled", pw: "difference_relu"}
    conn.close()
