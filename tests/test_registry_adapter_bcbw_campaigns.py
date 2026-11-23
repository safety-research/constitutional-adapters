"""The full BigCodeBench / BFCL readers PARAMETRIZED with a DECLARED CAMPAIGN (`_bcbw_common.CampaignSpec`, `TaskSpec.declared`; `DECLARED_CAMPAIGN_RULE` and the rules beside it), on
the SYNTHETIC campaign mirrors of `_bcbw_campaign_fixtures`: the production sikl4 declaration (`_bcbw_campaigns.SIKL4`; adapters `bigcodebench_full_sikl4` / `bfcl_full_sikl4`:
three views, six fitted cells each, no `none`, anchor `none`) and a synthetic campaign of the other shape (base-to-instruct fits, the view's own `none`, anchor
`own_none`). What it pins:

* THE DECLARATION: the sikl4 campaign binds exactly what the committed roster, pins and legacy spellings spell; a paper reader carries no campaign and its grammar
  objects are the module's own (the very objects);
* INGEST: per view, six runs on the full instrument and item set, run ids `si_target_refit:...` equal to an independent `identity.run_id`, the campaign column,
  object = pin through the roster's pin form, `substrate modal`, `transfer_kind instruct_direct` DERIVED, `extra.fit_seed` the view's, `seed_base` NULL, the value /
  bound / at-cap readings and the item rows; the runs reach `ingested` in a scratch registry's fold;
* THE ANCHOR MODES: `none` files no contrast, never asks the currency (the read passes with no connection), parks the VALUE row's delta block under
  `DECLARED_ANCHOR_RULE` with no refusal line; `own_none` files the view's own none as class anchor and pairs the view's rows against it, and with the own none not
  current parks every comparison by name -- never a borrowed contrast;
* REFUSALS BY NAME: a paper-grammar or an undeclared campaign-like row dir (the whole group), a suffix naming a paper launch tag, another view's tag or `-smoke`, a
  short outcome record, a `none` store the campaign does not admit, a prompted key, a launch row with a seed base, another fit seed or transfer kind, an object
  that is not the pin, a launched pins copy that disagrees, a base-checkpoint fit, a three-number record of another source, a launched entry of another wave, a
  roster pin form naming another pins file; the relaunch path parks the first attempt and reads `-r2`;
* ONE-TASK STORES: a row whose two tasks ran in two one-task stores (`-bcb` / `-bfcl`) is read from its task's store by each source, and the other task's store and
  gate record are neither discovered, read, parked nor refused (`suffix_task_for` reads the campaign's grammar, whose suffix spells the task tag);
* THE PAPER READERS UNTOUCHED: a campaign row dir in a paper mirror stays undeclared (refused whole), `run_payload_base` files the module's campaign by default
  (the emission pins are `test_registry_adapter_bcbw_full_emission_pin.py` / `test_registry_adapter_bcbw_emission_pin.py`);
* REGISTRATION: both sikl4 adapters registered, version 1, their owner, roots and sources; the dry read over a campaign mirror needs no registry connection."""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest
import yaml

import _anchor_frame_fixtures as AF
import _b1_fixtures as F
import _bcbw_full_fixtures as Z
import _bcbw_campaign_fixtures as LF
import _bcbw_seedrep_fixtures as X
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import _bcbw_campaigns as CS
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bfcl_full_sikl4 as BFL
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import bigcodebench_full_sikl4 as BCL
from sisteer.registry.adapters import lineage_common as LC

REPO = Path(__file__).resolve().parents[1]
EVAL = "bigcodebench"
SIKL4 = CS.SIKL4
V1, V2, V3 = SIKL4.entries
C = LF.SIKL4_CELLS
N = LF.N


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _sikl4(tmp_path, seeds, eval_id=EVAL, **kw):
    fx = LF.mirror(tmp_path, seeds, eval_id=eval_id, shape="sikl4", **kw)
    return fx, LF.adapter(fx, seeds, eval_id)


def _runs(bt):
    return {r["campaign_cell_id_legacy"]: r for r in bt.get("run", [])}


# --------------------------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------------------------

def test_the_sikl4_declaration_binds_what_the_committed_roster_pins_and_spellings_spell(seeds):
    assert (SIKL4.name, SIKL4.campaign, SIKL4.model_key, SIKL4.anchor, SIKL4.admits_none) == ("sikl4", "si_target_refit", "qwen35_4b", B.DECLARED_ANCHOR_NONE, False)
    assert SIKL4.harness_roster_rel == ("configs", "sdfbase_full22_roster_sikl4.yaml") and SIKL4.harness_pins_rel == ("configs", "sikl4_pins.json")
    assert SIKL4.launched_pins_file == "sikl4_pins.json" and SIKL4.producer_session == BCL.OWNER == BFL.OWNER
    roster = yaml.safe_load(open(REPO / "configs" / "sdfbase_full22_roster_sikl4.yaml", encoding="utf-8"))
    pins = json.load(open(REPO / "configs" / "sikl4_pins.json", encoding="utf-8"))
    views = roster["targets"]["qwen35_4b"]["views"]
    assert sorted(views) == sorted(v.selection for v in SIKL4.entries)
    for v in SIKL4.entries:
        assert (v.tag, v.row_dir, v.selection, v.transfer_kind) == (f"qwen35_4b_sikl4s{v.fit_seed}", f"qwen35_4b-f22sikl4s{v.fit_seed}", f"sikl4s{v.fit_seed}", "instruct_direct")
        cells = views[v.selection]["cells"]
        assert sorted(cells) == sorted(C)
        for okey, ent in cells.items():
            assert ent == {"pins": "configs/sikl4_pins.json", "pin": f"qwen35_4b.{v.selection}__{okey}"}, (v.selection, okey, ent)
            assert len(pins["qwen35_4b"][f"{v.selection}__{okey}"]["object_sha256"]) == 64
        mv = seeds.model_for_view_legacy(v.tag)
        assert (mv["model_id"], mv["transfer_kind"]) == ("qwen35_4b_it", v.transfer_kind), mv
    assert len(pins["qwen35_4b"]) == 18
    for mod, full in ((BCL, BCF), (BFL, BFF)):
        assert mod.SPEC.declared is SIKL4 and mod.SPEC.campaign == "si_target_refit" and mod.SPEC.models == ("qwen35_4b",)
        assert (mod.SPEC.instrument_seed, mod.SPEC.item_set_id, mod.SPEC.roster_file) == (full.SPEC.instrument_seed, full.SPEC.item_set_id, full.SPEC.roster_file)
        assert not (mod.SPEC.instruct_direct or mod.SPEC.unsteered_row or mod.SPEC.dose_relu_row or mod.SPEC.dose_lora_row or mod.SPEC.lecun_relu_row)
        assert mod.SPEC.pins_file == "sikl4_pins.json" and mod.SPEC.three_numbers_file == f"three_numbers_{mod.SOURCE}.json"
        # the paper reader carries no campaign; its grammar objects are the module's own
        assert full.SPEC.declared is None and full.SPEC.suffix_re is B.SUFFIX_RE and full.SPEC.gate_re is B.GATE_RE and full.SPEC.pins_variant_re is B.PINS_VARIANT_RE
        assert full.SPEC.campaign == B.CAMPAIGN == LC.CAMPAIGN == "sdfbase_full22" and full.SPEC.pins_file == B.PINS_FILE == "sdfbase_full22_pins.json"


def test_the_campaign_grammar_is_the_declared_table_matched_exactly():
    spec = BCL.SPEC
    sha7 = "4c11068"
    for v in SIKL4.entries:
        assert spec.suffix_re.match(f"bcbw-{v.tag}-{sha7}") and spec.suffix_re.match(f"bcbw-{v.tag}-{sha7}-r2") and spec.suffix_re.match(f"bcbw-{v.tag}-{sha7}-r2-bcb")
        assert B.tag_of_row_dir(v.row_dir, spec) == v.tag and B.group_of(spec, v.tag) == v.tag and B.model_key_for(spec, v.tag) == "qwen35_4b"
        assert B.tags_of(spec, v.tag) == {v.selection: v.tag} and B.row_dir_for(spec, v.tag) == v.row_dir
        assert B.gate_name_parts_for(spec, f"gate_{v.tag}_{C[0]}_bcbw-{v.tag}-{sha7}.json") == {"tag": v.tag, "okey": C[0], "suffix": f"bcbw-{v.tag}-{sha7}"}
    # the task tag (ONE_TASK_STORE_RULE) under the campaign's own grammar: a paper reader's answer is suffix_task's, a campaign reader's its declared suffix's
    for v in SIKL4.entries:
        for tail, task in (("", None), ("-bcb", "bigcodebench"), ("-bfcl", "bfcl"), ("-r2-bcb", "bigcodebench"), ("-r3-bfcl", "bfcl"), ("-r2", None)):
            assert B.suffix_task_for(spec, f"bcbw-{v.tag}-{sha7}{tail}") == task and B.suffix_task_for(BFL.SPEC, f"bcbw-{v.tag}-{sha7}{tail}") == task, (v.tag, tail)
            assert B.suffix_task_for(BCF.SPEC, f"bcbw-{v.tag}-{sha7}{tail}") is None is B.suffix_task(f"bcbw-{v.tag}-{sha7}{tail}")     # outside the paper grammar
    for paper in (f"bcbw-qwen35_4b_s1-{sha7}-bcb", f"bcbw-qwen35_4b_it-{sha7}-r2-bfcl", f"bcbw-qwen35_4b_s1-{sha7}"):
        assert B.suffix_task_for(BCF.SPEC, paper) == B.suffix_task(paper) and B.suffix_task_for(spec, paper) is None, paper
    for bad in (f"bcbw-qwen35_4b_sikl4s1-{sha7}-smoke", f"bcbw-qwen35_4b_sikl4s1-{sha7}-w8192", f"bcbw-qwen35_4b_s1-{sha7}", f"bcbw-qwen35_4b_sikl4s1_arditi-{sha7}",
                f"bcbw-qwen35_4b_sikl4s1-{sha7}-sibat", f"bcbw-qwen35_4b_sikl4s4-{sha7}", f"bcbw-qwen35_4b_sikl4s1-{sha7}-shared"):
        assert spec.suffix_re.match(bad) is None, bad
    for row_dir in ("qwen35_4b-f22s1", "qwen35_4b-f22sikl4s4", "qwen35_4b-f22it", "gemma4e4b-f22s1"):
        assert B.tag_of_row_dir(row_dir, spec) is None and B.is_undeclared_row_dir_group(row_dir, spec)
        assert "is not a row of the declared campaign 'sikl4'" in B.undeclared_row_dir_reason(row_dir, spec)
    # the paper grammar does not read a campaign row dir: it stays undeclared for a paper reader, as before
    assert B.tag_of_row_dir(V1.row_dir, BCF.SPEC) is None and B.tag_of_row_dir(V1.row_dir) is None and "no declared row" in B.undeclared_row_dir_reason(V1.row_dir, BCF.SPEC)
    assert B.group_of(BCF.SPEC, "qwen35_4b_s2p") == "qwen35_4b_s2" == B.seed_tag_of("qwen35_4b_s2p") and B.model_key_for(BCF.SPEC, V1.tag) == B.model_key_of(V1.tag) is None
    with pytest.raises(ValueError):
        dataclasses.replace(SIKL4, anchor="borrowed")
    with pytest.raises(ValueError):
        dataclasses.replace(SIKL4, anchor=B.DECLARED_ANCHOR_OWN_NONE)        # an own_none anchor needs the campaign to admit its none


# --------------------------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("eval_id", ["bigcodebench", "bfcl"])
def test_every_view_files_its_six_fitted_runs_under_the_declared_campaign_with_no_registry_connection(tmp_path, seeds, eval_id):
    fx, ad = _sikl4(tmp_path, seeds, eval_id=eval_id)
    out = LF.read_all(ad, conn=None)            # anchor `none`: the currency is never asked, so a read with no registry connection passes
    assert sorted(out) == sorted(SIKL4.tag_names)
    mod = BCL if eval_id == "bigcodebench" else BFL
    inst = str(ad.seeds.instrument_row(mod.SPEC.instrument_seed, eval_id, model_id="qwen35_4b_it")["instrument_id"])
    comp_id = mod.SPEC.recount_computation_id()
    for v in SIKL4.entries:
        bt, facts = out[v.tag]
        assert facts["runs"] == 6 and facts["n_refusals"] == 0 and not facts.get("refuse") and facts["declared_campaign"]["view"] == v.tag and facts["contrasts_declared_none"] == 6
        assert "run_contrast" not in bt and facts["contrasts_filed"] == 0 and facts["group_rule"] == B.DECLARED_GRAMMAR_RULE
        runs = _runs(bt)
        assert sorted(runs) == sorted(C)
        for okey, r in runs.items():
            skey = fx["keys"][f"{v.tag}|{okey}"]
            assert r["run_id"] == I.run_id("si_target_refit", f"{v.tag}/{skey}", okey, eval_id, inst) and r["run_id"].startswith(f"si_target_refit:{eval_id}:")
            assert (r["campaign"], r["view_legacy"], r["lane_source_legacy"], r["model_id"], r["transfer_kind"]) == ("si_target_refit", v.tag, mod.SOURCE, "qwen35_4b_it", "instruct_direct")
            assert r["object_id"] == LF.sha_of(SIKL4, v, okey) and r["substrate"] == "modal" and r["seed_base"] is None and r["trial_index"] == 0
            assert r["item_set_id"] == mod.SPEC.item_set_id and r["instrument_id"] == inst and r["producer_session"] == SIKL4.producer_session
            assert r["field_confidence_json"]["transfer_kind"]["confidence"] == "derived" and "derive.transfer_kind" in r["field_confidence_json"]["transfer_kind"]["source"]
            ex = r["extra"]
            assert ex["fit_seed"] == v.fit_seed and ex["declared_campaign"]["name"] == "sikl4" and ex["declared_campaign"]["selection"] == v.selection and ex["declared_campaign"]["transfer_kind_derived"] == "instruct_direct"
            assert ex["anchor"]["state"] == B.DECLARED_NONE_STATE and ex["anchor_rule"] == B.DECLARED_ANCHOR_RULE and "appendix_family" not in ex
            assert ex["launched_pins_record"].endswith(os.path.join("records", "sikl4_pins.json"))
            metrics = [m for m in bt["run_metric"] if m["run_id"] == r["run_id"]]
            value = [m for m in metrics if m["computation_id"] == comp_id]
            bound = [m for m in metrics if m["computation_id"] == ""]
            sp = fx["specs"][f"{v.tag}|{okey}"][eval_id]
            c = X.counts(sp)
            assert len(value) == 1 and len(bound) == 1 and value[0]["k"] == c["correct"] and value[0]["n"] == c["n_scored"] and bound[0]["n"] == N
            assert value[0]["extra"]["delta_block"]["rule"] == B.DECLARED_ANCHOR_RULE and value[0]["extra"]["delta_block"]["anchor"]["state"] == B.DECLARED_NONE_STATE
            assert sum(1 for it in bt["item_result"] if it["run_id"] == r["run_id"]) == N
        assert not [x for x in facts["refusals"] if "|contrast" in x], "a declared-none park files no refusal line"


def test_the_source_states_name_the_views_launch_rows_and_the_campaign_pins_the_roster_references(tmp_path, seeds):
    fx, ad = _sikl4(tmp_path, seeds, views=(1,))
    inputs = ad.discover()
    groups = LF.groups_of(ad)
    for g in groups:
        list(ad.read(X.ctx(), g, groups[g]))
    states = {(s.input_uri.rsplit("/", 1)[1]): s for s in ad.source_states(X.ctx(), inputs)}
    pins = states[f"sikl4_pins.json#{V1.tag}"]
    assert pins.json_path == "$.qwen35_4b" and pins.current_members == sorted(f"{V1.selection}__{c}" for c in C)
    ls = states[f"launch_sets_full.json#{V1.tag}"]
    assert ls.current_members == sorted(f"{V1.selection}/{c}" for c in C)
    tn = states[f"three_numbers_bigcodebench_full_sikl4.json#{V1.tag}"]
    assert tn.current_members == sorted(f"{V1.tag}/{c}" for c in C)


def test_campaign_runs_reach_ingested_in_a_scratch_registry_fold(tmp_path, seeds):
    from sisteer.registry.adapters import dimensions as DIM
    fx, ad = _sikl4(tmp_path, seeds, views=(1, 2))
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    roles = {"safety_refusal": "safety", "si_constitution": "constitution", "C16": "irrelevant"}      # the live object rows' roles (the reader files no object record)
    for v in fx["views"]:
        for okey in C:
            obj = ad.resolve_object(okey, v.selection, model_key="qwen35_4b")[0]
            # (the dimension seeds carry no C16 text -- the candidate SI's text stays off the repo -- so its stand-in object row names another declared teacher;
            # a test-only placeholder: the reader files no object record and the fold only needs the object id to exist)
            obj = dataclasses.replace(obj, object_role=roles[obj.si_id], si_id=(obj.si_id if obj.si_id != "C16" else "si_constitution"))
            r = LC.standin_object_record(obj, seeds, F.producer(), now, "qwen35_4b_it")
            assert r is not None, okey
            if r:
                recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    rep1, ing1, ing1b = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep1.groups_refused == 0 and ing1.refused + ing1b.refused == 0, ([r.error for r in rep1.results], ing1.refusals, ing1b.refusals)
    rows = F.q(root, "SELECT campaign, view_legacy, lifecycle_state, n_items_planned, decode_policy_match, COUNT(*) FROM run WHERE lane_source_legacy = ? GROUP BY 1, 2, 3, 4, 5 ORDER BY 2",
               (BCL.SOURCE,))
    assert rows == [("si_target_refit", V1.tag, "ingested", N, "exact", 6), ("si_target_refit", V2.tag, "ingested", N, "exact", 6)], rows
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.lane_source_legacy = ?", (BCL.SOURCE,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ?", (BCL.SOURCE,)) == [(12 * N,)]
    assert F.q(root, "SELECT MIN(ok), COUNT(*) FROM v_source_status_current WHERE adapter = ?", (BCL.NAME,)) == [(1, 2)]


# --------------------------------------------------------------------------------------------
# the anchor modes
# --------------------------------------------------------------------------------------------

def test_an_own_none_campaign_files_its_own_none_as_the_anchor_and_pairs_the_views_rows_against_it(tmp_path, seeds):
    fx = LF.mirror(tmp_path, seeds, eval_id=EVAL, shape="ctlstop", views=(1, 2))
    ad = LF.adapter(fx, seeds, EVAL)
    out = LF.read_all(ad, primed=True)
    for v in fx["views"]:
        bt, facts = out[v.tag]
        assert facts["runs"] == 5 and facts["n_refusals"] == 0 and facts["contrasts_filed"] == 4 and facts["contrasts_own"] == 4 and facts["contrasts_reanchored"] == 0
        runs = _runs(bt)
        none = runs[LF.NONE]
        assert none["campaign"] == "reseed_ctlstop" and none["extra"]["launch_set"]["class"] == "anchor" and none["extra"]["fit_seed"] is None
        assert none["extra"]["same_object_rows"]["twin_tags_legacy"] == [t for t in LF.CTLSTOP.tag_names if t != v.tag]
        for okey in LF.CTLSTOP_CELLS:
            r = runs[okey]
            assert r["transfer_kind"] == "base_to_instruct" and r["extra"]["fit_seed"] == v.fit_seed and r["extra"]["anchor"]["anchor_run_id"] == none["run_id"]
            assert r["extra"]["anchor"]["rule"] == B.DECLARED_ANCHOR_RULE
        assert {(c["run_id"], c["anchor_run_id"], c["anchor_kind"]) for c in bt["run_contrast"]} == {(runs[o]["run_id"], none["run_id"], "same_standup_none") for o in LF.CTLSTOP_CELLS}
        assert all(c["extra"]["anchor_rule"] == B.DECLARED_ANCHOR_RULE and c["n_pairs"] > 0 and c["computed_by"] == "registry" for c in bt["run_contrast"])


def test_an_own_none_campaign_parks_by_name_when_its_own_none_is_not_current_and_never_borrows(tmp_path, seeds):
    fx = LF.mirror(tmp_path, seeds, eval_id=EVAL, shape="ctlstop", views=(1, 2))
    ad = LF.adapter(fx, seeds, EVAL)
    frame = X.primed_frame(ad)
    first = LF.read_all(ad, conn=frame.con)
    own = _runs(first[LF.CTLSTOP.entries[0].tag][0])[LF.NONE]["run_id"]
    AF.retire(frame, own)            # the view's own none retired: the currency's exact-key search re-picks the other view's none (a sibling standup of this source)
    out = LF.read_all(ad, conn=frame.con)
    bt, facts = out[LF.CTLSTOP.entries[0].tag]
    assert "run_contrast" not in bt and facts["contrasts_filed"] == 0 and facts["contrasts_parked"] == 4, facts["contrasts"]
    for okey in LF.CTLSTOP_CELLS:
        r = _runs(bt)[okey]
        blk = r["extra"]["anchor"]
        # the currency re-picked the other view's none (a same-source sibling standup) and the campaign refused the pick: a park, never a borrowed contrast
        assert blk["state"] == "no_current_anchor" and blk["rule"] == B.DECLARED_ANCHOR_RULE and blk["anchor_run_id"] is None, blk
        assert blk["currency_state"] not in ("own_current", "no_current_anchor") and "parks rather than borrowing" in blk["reason"], blk
        line = LF.refusal(facts, LF.CTLSTOP.entries[0].tag, f"{okey}|contrast")
        assert "filed without a contrast" in line
    # the other view still pairs against its own current none
    assert out[LF.CTLSTOP.entries[1].tag][1]["contrasts_own"] == 4


# --------------------------------------------------------------------------------------------
# refusals by name
# --------------------------------------------------------------------------------------------

def test_a_paper_grammar_or_an_undeclared_campaign_like_row_dir_refuses_its_whole_group_by_name(tmp_path, seeds):
    fx, ad = _sikl4(tmp_path, seeds, views=(1,), paper_row_dir=True, undeclared_row_dir="qwen35_4b-f22sikl4s4")
    out = LF.read_all(ad)
    assert sorted(out) == sorted([V1.tag, "qwen35_4b-f22s1", "qwen35_4b-f22sikl4s4"])
    for g in ("qwen35_4b-f22s1", "qwen35_4b-f22sikl4s4"):
        bt, facts = out[g]
        assert not bt and facts["runs"] == 0 and len(facts["refuse"]) == 1 and "is not a row of the declared campaign 'sikl4'" in facts["refuse"][0]
    assert out[V1.tag][1]["runs"] == 6 and out[V1.tag][1]["n_refusals"] == 0


def test_store_and_record_defects_refuse_their_row_by_name_and_the_views_other_rows_file(tmp_path, seeds):
    sha7 = LF.TREE[:7]
    other = "e" * 64
    fx, ad = _sikl4(tmp_path, seeds, views=(1, 2, 3),
                    suffix_override={(1, C[0]): f"bcbw-qwen35_4b_s1-{sha7}", (1, C[1]): f"bcbw-{V1.tag}-{sha7}-smoke", (1, C[2]): f"bcbw-{V2.tag}-{sha7}"},
                    short_outcomes={(1, C[3])},
                    launch_row_edit={(2, C[0]): lambda r: r.update(seed_base=1), (2, C[1]): lambda r: r.update(fit_seed=1), (2, C[2]): lambda r: r.update(transfer_kind="base_to_instruct")},
                    attest_sha={(2, C[3]): other}, launched_pins_sha={(3, C[0]): other}, lane_roster_wave={(3, C[1]): "qwen35_4b_s1"})
    out = LF.read_all(ad)
    f1, f2, f3 = (out[v.tag][1] for v in (V1, V2, V3))
    assert (f1["runs"], f1["n_refusals"], f2["runs"], f2["n_refusals"], f3["runs"], f3["n_refusals"]) == (2, 4, 2, 4, 4, 2), (f1["refusals"], f2["refusals"], f3["refusals"])
    assert "outside the grammar" in LF.refusal(f1, V1.tag, C[0]) and "outside the grammar" in LF.refusal(f1, V1.tag, C[1])
    assert f"does not spell this row's launch tag '{V1.tag}'" in LF.refusal(f1, V1.tag, C[2])
    assert f"holds {N - 1} items, not {N}" in LF.refusal(f1, V1.tag, C[3])
    assert "seed_base 1" in LF.refusal(f2, V2.tag, C[0]) and B.DECLARED_GRAMMAR_RULE[:40] in LF.refusal(f2, V2.tag, C[0])
    assert "fit_seed 1 (the view declares 2)" in LF.refusal(f2, V2.tag, C[1])
    assert "transfer_kind 'base_to_instruct' (the view declares instruct_direct)" in LF.refusal(f2, V2.tag, C[2])
    assert "is no prefix of the pinned object" in LF.refusal(f2, V2.tag, C[3])
    assert "the launched pins (records/sikl4_pins.json) name 'eeeeeeeeeeeeeeee'" in LF.refusal(f3, V3.tag, C[0])
    assert f"names launch tag 'qwen35_4b_s1', not '{V3.tag}'" in LF.refusal(f3, V3.tag, C[1])


def test_a_none_or_a_prompted_store_under_a_campaign_that_admits_neither_refuses_by_name(tmp_path, seeds):
    fx, ad = _sikl4(tmp_path, seeds, views=(1,), none_store=True, prompted_store=True)
    bt, facts = LF.read_all(ad)[V1.tag]
    assert facts["runs"] == 6 and facts["n_refusals"] == 2, facts["refusals"]
    assert "the campaign admits no none" in LF.refusal(facts, V1.tag, LF.NONE)
    assert "a prompted key under the declared campaign 'sikl4'" in LF.refusal(facts, V1.tag, LF.PROMPTED)


def test_a_base_checkpoint_fit_a_foreign_three_number_record_and_a_foreign_pin_form_refuse_by_name(tmp_path, seeds):
    _fx, ad = _sikl4(tmp_path / "a", seeds, views=(1,), fit_hf=LF.BASE_HF)
    facts = LF.read_all(ad)[V1.tag][1]
    assert facts["runs"] == 0 and facts["n_refusals"] == 6 and all("transfer_kind: derived 'base_to_instruct'" in x and "files only instruct_direct" in x for x in facts["refusals"])
    _fx, ad = _sikl4(tmp_path / "b", seeds, views=(1,), three_source="bigcodebench_full")
    facts = LF.read_all(ad)[V1.tag][1]
    assert facts["runs"] == 0 and facts["n_refusals"] == 6 and all("spells source 'bigcodebench_full'" in x for x in facts["refusals"])
    _fx, ad = _sikl4(tmp_path / "c", seeds, views=(1,), harness_pins_rel_override="configs/sdfbase_full22_pins.json")
    facts = LF.read_all(ad)[V1.tag][1]
    assert facts["runs"] == 0 and facts["n_refusals"] == 6 and all("names pins file 'configs/sdfbase_full22_pins.json'" in x and "the campaign declares configs/sikl4_pins.json" in x
                                                                     for x in facts["refusals"])


def test_a_relaunched_campaign_row_reads_its_r2_store_and_parks_the_first_attempt(tmp_path, seeds):
    fx, ad = _sikl4(tmp_path, seeds, views=(1,), relaunched={(1, C[0])})
    bt, facts = LF.read_all(ad)[V1.tag]
    assert facts["runs"] == 6 and facts["n_refusals"] == 0 and facts["n_parked"] == 1 and facts["parked"][0]["object_key"] == C[0], (facts["refusals"], facts.get("parked"))
    r = _runs(bt)[C[0]]
    assert r["code_sha"] == LF.TREE2 and r["extra"]["store_suffix_legacy"].endswith("-r2") and r["extra"]["launch_set"]["relaunch"]["tag"] == "r2"


@pytest.mark.parametrize("eval_id", ["bigcodebench", "bfcl"])
def test_the_other_tasks_one_task_store_and_its_gate_record_are_invisible_to_a_campaign_reader(tmp_path, seeds, eval_id):
    # ONE_TASK_STORE_RULE under the campaign grammar (a one-task relaunch, `--tag r<N>-bcb`, writes such a store): the row's two tasks ran in two ONE-TASK stores,
    # each named by its eval's launched_as; each source reads ITS task's store, and the other task's store and its gate record are not this source's -- neither
    # discovered, read, parked nor refused. Until suffix_task_for read the campaign's grammar it fell to the paper SUFFIX_RE, matched no campaign suffix, and so
    # bound the other task's store to the row (a parked or refused line where the census expects none)
    sha7 = LF.TREE[:7]
    fx, ad = _sikl4(tmp_path, seeds, eval_id=eval_id, views=(1,), one_task_stores={(1, C[0])})
    own_tag, other_tag, other_eval = ("bcb", "bfcl", "bfcl") if eval_id == "bigcodebench" else ("bfcl", "bcb", "bigcodebench")
    own, other = f"bcbw-{V1.tag}-{sha7}-{own_tag}", f"bcbw-{V1.tag}-{sha7}-{other_tag}"
    store_dir = os.path.join(fx["root"], B.STORES_HEAD, V1.row_dir, C[0], B.THINKING_DIR)
    assert sorted(os.listdir(store_dir)) == sorted([own, other])
    assert os.path.isfile(os.path.join(fx["root"], B.RECORDS_DIR, f"gate_{V1.tag}_{C[0]}_{other}.json"))
    assert B.suffix_task_for(ad.SPEC, own) == eval_id and B.suffix_task_for(ad.SPEC, other) == other_eval
    rels = [os.path.relpath(i.uri.rsplit("#", 1)[0], fx["root"]) for i in ad.discover()]
    assert not [r for r in rels if other in r], "neither the other task's store files nor its gate record are this source's inputs"
    assert [r for r in rels if os.sep + own + os.sep in r] and os.path.join(B.RECORDS_DIR, f"gate_{V1.tag}_{C[0]}_{own}.json") in rels
    bt, facts = LF.read_all(ad)[V1.tag]
    assert facts["runs"] == 6 and facts["n_refusals"] == 0 and "parked" not in facts and not facts.get("not_run"), (facts["refusals"], facts.get("parked"))
    assert not [g for g in facts.get("gate_records_without_store", []) if other in str(g.get("suffix"))]
    r = _runs(bt)[C[0]]
    assert r["extra"]["store_suffix_legacy"] == own and r["code_sha"] == LF.TREE and r["object_id"] == LF.sha_of(SIKL4, V1, C[0])
    sp = fx["specs"][f"{V1.tag}|{C[0]}"][eval_id]
    assert sum(1 for it in bt["item_result"] if it["run_id"] == r["run_id"]) == N and X.counts(sp)["correct"] == [
        m for m in bt["run_metric"] if m["run_id"] == r["run_id"] and m["computation_id"] == ad.SPEC.recount_computation_id()][0]["k"]


# --------------------------------------------------------------------------------------------
# the paper readers untouched
# --------------------------------------------------------------------------------------------

def test_a_campaign_row_dir_in_a_paper_mirror_stays_undeclared_and_run_payload_base_files_the_modules_campaign(tmp_path, seeds):
    fx = Z.mirror(tmp_path, seeds, eval_id=EVAL, root=str(tmp_path / "paper"), undeclared_row_dirs=(V1.row_dir,))
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    groups = LF.groups_of(ad)
    assert V1.row_dir in groups
    conn = X.primed_conn(ad)
    for g in sorted(groups):
        recs = [r for b in ad.read(X.ctx(conn), g, groups[g]) for r in b.records]
        facts = ad.status_extra(g)
        if g == V1.row_dir:
            assert not recs and "names the declared model key 'qwen35_4b' but no declared row" in facts["refuse"][0]
        else:
            assert {r["payload"]["campaign"] for r in recs if r["record_type"] == "run"} == {"sdfbase_full22"}
    assert inspect.signature(LC.run_payload_base).parameters["campaign"].default == LC.CAMPAIGN == "sdfbase_full22"


# --------------------------------------------------------------------------------------------
# registration, and the dry read over a campaign mirror
# --------------------------------------------------------------------------------------------

def test_both_sikl4_adapters_are_registered_with_version_one_their_owner_roots_and_sources():
    from sisteer.registry import adapters as ADP
    reg = ADP.load_all()
    for mod, cls in ((BCL, BCL.BigcodebenchFullSikl4Adapter), (BFL, BFL.BfclFullSikl4Adapter)):
        inst = reg[mod.NAME]()
        assert type(inst) is cls and inst.covers_sources == (mod.SOURCE,) and inst.owner == SIKL4.producer_session
        assert AB.emission_version_of(cls) == "1" and inst.roots == (os.path.join(LC.RS_DEFAULT, mod.SOURCE),) == (mod.ROOT_DEFAULT,)
        assert inst.roster_path.endswith(os.path.join("configs", "sdfbase_full22_roster_sikl4.yaml")) and os.path.isfile(inst.roster_path)
        assert inst.pins_path.endswith(os.path.join("configs", "sikl4_pins.json")) and inst.shared_files()[2] == "sikl4_pins.json"
    with pytest.raises(ValueError):
        type("Both", (B.BcbwSeedrepReader,), {"name": "both_x", "SPEC": dataclasses.replace(BCL.SPEC, shared=B.SharedSpec("x", "x"))})()


def test_the_dry_read_over_a_campaign_mirror_needs_no_registry_connection(tmp_path, seeds, monkeypatch, capsys):
    script = REPO / "scripts" / "registry" / "bcbw_dry_read.py"
    spec = importlib.util.spec_from_file_location("bcbw_dry_read_campaign_test", script)
    dry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dry)
    fx, _ad = _sikl4(tmp_path, seeds)
    fseeds = Z.seeds_for(seeds)

    class FixtureLane(BCL.BigcodebenchFullSikl4Adapter):
        def __init__(self, root=None, seeds=None, **_kw):
            super().__init__(root=root or fx["root"], seeds=fseeds, roster_path=fx["roster"], pins_path=fx["pins"], pins_root=fx["checkout"])

    from sisteer.registry import adapters as ADP
    monkeypatch.setattr(ADP, "load_all", lambda names=None: {BCL.NAME: FixtureLane})
    out = tmp_path / "dry"
    assert dry.main(["--adapter", BCL.NAME, "--root", fx["root"], "--out", str(out), "--no-record-id"]) == 0
    capsys.readouterr()
    summary = json.loads((out / "summary.json").read_text())
    assert sorted(summary["groups"]) == sorted(SIKL4.tag_names)
    for g, s in summary["groups"].items():
        assert s["runs"] == 6 and s["n_refusals"] == 0 and s["malformed"] == [] and "run_contrast" not in s["by_type"], (g, s)
