"""The SHARED-STANDUP comparison readers (`bigcodebench_full_shared` / `bfcl_full_shared`, 2026-09-28) -- the solo full readers PARAMETRIZED
(`_bcbw_common.SharedSpec` on `TaskSpec`) -- on the SYNTHETIC shared mirror of `_bcbw_shared_fixtures` (derived from the synthetic full mirror the way the
fixture builder derives the real fixture from the canonical solo stores) and, where the machine carries it, on the real synthetic fixture under
`FIXTURE_ROOT` (read-only; skipped elsewhere). What it pins:

* THE GRAMMAR: `bcbw-<launch tag>-<sha7>-shared[-r<N>][-rep<N>]` on the shared specs alone; the solo specs carry `shared` None and their `suffix_re` IS the
  module's `SUFFIX_RE`, which admits none of the shared suffixes;
* THE READ: every store files on the `_shared` spelling (`view_legacy gemma4e4b_s1_shared` / `gemma4e4b_it_shared`), the producer key `<tag>_shared/<store key>[@rep<N>]`,
  `extra.comparison_kind` / `solo_run_id` / `solo_store` / `standup_id` / `row_engine` on every run, the in-run contrast against the row's shared none
  (`same_standup_none`; `sibling_none` for the relaunch on its own standup) AND the solo stand-in contrast (anchor_kind `solo_stand_in`) against the SOLO run read from the solo mirror (both
  values, the solo id the SOLO reader mints), the parked first attempt, the instruct row's own none as a comparison row that anchors nothing;
* THE EIGHT IDENTITY HAZARDS of REQUIRED_RECORDS.md section 6, one test each: (1) run ids under the `_shared` spelling, never the solo tag's, and a
  solo_run_id that is not the solo reader's refuses; (2) the repeat row keyed by its segment, two rows on one key refuse; (3) `view` the registry spelling,
  `selection` the key -- the solo spelling under `view` refuses; (4) `lane_roster.json:wave` the launch tag -- the `_shared` spelling there refuses; (5) the it none's
  cross-tree roster citation matched by hash; (6) two producer trees in one row (the real fixture), `cross_tree` recorded; (7) the parked `-shared` store mirrored
  COMPLETE parks, never reads, never refuses; (8) the frame: `v_roster_match` ignores the spelling and the source, so a shared run a roster entry matches reads
  `selected` (and its solo twin stays `selected` through the include) until an `exclude` scoped to the shared sources covers it -- the "not_in_roster by
  construction" claim holds only for rows no entry matches, and the remedy is a ruling, filed by the registry's owner, never a reader default;
* REFUSED BY NAME: a RESULT `standup=` token that is not the manifest's, a `row_engine` block with another key set, a gate `comparability_class` disagreeing with its
  standup ids; the comparison WAITING (`solo_store_absent`) when the solo mirror lacks the store, the run filed;
* THE STORE-ATTESTATION DECISION of 2026-09-29 (the first REAL mirror, landing 1; `_bcbw_common.SHARED_ROW_DIR_RULE` / `SHARED_ENDPOINT_TUPLE_RULE` /
  `SHARED_IT_REPEAT_RULE`), one test each: (1) `row_engine.row_dir` volume-absolute (`/vol/tinycap/gen/rows/<row>/<suffix>`) binds by its `rows/<row>/<suffix>` tail -- the
  row launch's container: the row's primary launch tag, the store's relaunch segment, no repeat segment -- and is recorded verbatim; another row, another launch tag, a
  repeat segment or a container the launch row does not declare refuses; (2) the served-digest map is read from `served_geometry.json:endpoint` when the manifest's
  route copy is null, the store's tuple is the manifest's, and an endpoint whose `cell` / `harness_cell` name the row's LAST served key (landing 1's row runner) reads
  with `object_id` downgraded to derived and a status note, while an agreeing endpoint reads clean with no note; a map entry that is not the pinned object, an
  endpoint naming any other key, a missing map or two unequal maps refuse; (3) the instruct row's own none served a second time (`-shared-rep2` under the `_it` row)
  is a repeat_floor row -- class repeat, no solo run, no solo store (`no_solo_twin`), its in-run floor against the row's own none -- never refused and never filed
  as it_none_comparison;
* registration, source, owner, version 2 (the bump that re-reads the accepted zero-run version-1 revisions), the spellings in legacy_spellings.yaml;
* LANDING 2 (`_l2`) and LANDING 3 (`_l3`, the landing-3 harness) -- the same readers parametrized once more each, their ids disjoint from every earlier
  landing's over one store key, the verified store's standup rule, the three-number record naming the source, and landing 3's three shapes (the fourteen-key verified
  block, the token-present re-stamp, the non-null legacy route scalar that never makes an unsteered or prompted store fitted), all read on every landing's reader.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest
import yaml

import _b1_fixtures as F
import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
import _anchor_frame_fixtures as AF
import _bcbw_shared_fixtures as SH
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import site_forms as SF
from sisteer.registry import synth
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bfcl_full_shared as BFS
from sisteer.capeval.tinycap import contracts as C
from sisteer.registry.adapters import bfcl_full_shared_l2 as BFS2
from sisteer.registry.adapters import bfcl_full_shared_l3 as BFS3
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import bigcodebench_full_shared as BCS
from sisteer.registry.adapters import bigcodebench_full_shared_l2 as BCS2
from sisteer.registry.adapters import bigcodebench_full_shared_l3 as BCS3
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import roster as RO

TAG, IT_TAG, NONE, RELU, PROMPTED = SH.TAG, SH.IT_TAG, SH.NONE, SH.RELU, SH.PROMPTED
GROUP = TAG
#: the synthetic shared-standup fixture sits under the site's work directory ($WORK): empty where this host's site names none
FIXTURE_ROOT = os.path.expanduser(SF.expansion("$WORK/tinycap-shared-standup/fixtures/run-state/full22") or "")
SOLO_LIVE_ROOT = LC.RS_DEFAULT
REAL_FIXTURE = pytest.mark.skipif(not FIXTURE_ROOT or not os.path.isdir(os.path.join(FIXTURE_ROOT, "bigcodebench_full_shared"))
                                  or not os.path.isdir(os.path.join(SOLO_LIVE_ROOT, "bigcodebench_full")),
                                  reason="the synthetic fixture (under the site's work directory) and the live solo mirrors live on the registry's machine only")


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _read(tmp_path, seeds, eval_id="bigcodebench", **kw):
    fx = SH.mirror(tmp_path, seeds, eval_id=eval_id, **kw)
    ad = SH.adapter(eval_id, fx, seeds)
    out, inputs = SH.read_all(ad)
    bt, facts = out[GROUP]
    return fx, ad, bt, facts


def _contrasts(bt, kind=None):
    return [c for c in bt.get("run_contrast", []) if kind is None or c["anchor_kind"] == kind]


def _by_view_key(bt):
    """The runs keyed ``(view_legacy, row key)``: the instruct row's none and the seed row's none both carry the row key ``none``."""
    return {(r["view_legacy"], r["extra"]["shared_standup"]["row_key"]): r for r in bt.get("run", [])}


def _refusal(facts, key: str) -> str:
    hits = [x for x in facts["refusals"] if x.startswith(f"{key} -- ")]
    assert len(hits) == 1, (key, facts["refusals"])
    return hits[0]


# --------------------------------------------------------------------------------------------
# the specs and the grammar
# --------------------------------------------------------------------------------------------

def test_the_shared_specs_bind_the_grammar_and_the_solo_specs_carry_none_of_it():
    for mod, solo in ((BCS, BCF), (BFS, BFF)):
        s = mod.SPEC
        assert s.shared is not None and s.shared.solo_source == solo.SOURCE and s.shared.solo_root_default.endswith(os.path.join("run-state", "full22", solo.SOURCE))
        assert s.source == mod.SOURCE == mod.NAME == f"{solo.SOURCE}_shared" and s.root_default.endswith(os.path.join("run-state", "full22", mod.SOURCE)) and s.root_default != solo.SPEC.root_default
        assert s.launch_set_files == ("launch_sets_shared.json",) and s.schedule_file == "bcbw_schedule_shared.json" and s.three_numbers_file == f"three_numbers_{solo.SOURCE}_shared.json"
        # everything else is the solo spec's: the eval, the task, the item set and instrument, the roster file, the model set, the instruct-direct admission
        for f in ("eval_id", "task", "record_task", "outcomes_file", "instrument_seed", "cap", "protocol_temperature", "item_set_key", "item_set_id", "item_set_label", "item_set_pin_file",
                  "roster_file", "models", "instruct_direct", "clamp_override", "tool_calls_required", "n_items"):
            assert getattr(s, f) == getattr(solo.SPEC, f), f
        assert s.spelling_of("gemma4e4b_s1") == "gemma4e4b_s1_shared" and s.spelling_of("qwen35_4b_it") == "qwen35_4b_it_shared" and mod.spelling_of("x_s1") == "x_s1_shared"
        for sfx, groups in (("bcbw-gemma4e4b_s1-8ba0b48-shared", {"relaunch": None, "repeat": None}), ("bcbw-gemma4e4b_s1-8ba0b48-shared-rep2", {"relaunch": None, "repeat": "-rep2"}),
                            ("bcbw-gemma4e4b_s1-f1cd7f2-shared-r2", {"relaunch": "-r2", "repeat": None}), ("bcbw-qwen35_4b_it-8ba0b48-shared", {"relaunch": None, "repeat": None}),
                            ("bcbw-gemma4e4b_s1_arditi-8ba0b48-shared", {"relaunch": None, "repeat": None}), ("bcbw-gemma4e4b_s1-8ba0b48-shared-r3-rep2-bcb", {"relaunch": "-r3", "repeat": "-rep2"})):
            m = s.suffix_re.match(sfx)
            assert m is not None and {k: m.group(k) for k in groups} == groups and m.group("marker") == "shared", sfx
            assert B.SUFFIX_RE.match(sfx) is None, "the solo grammar admits no shared suffix"
        for sfx in ("bcbw-gemma4e4b_s1-8ba0b48", "bcbw-gemma4e4b_s1-8ba0b48-r2", "bcbw-gemma4e4b_s1-8ba0b48-shared-negctl", "bcbw-gemma4e4b_s1-8ba0b48-rep2-shared", "bcbw-gemma4e4b_s1-8ba0b48-sharedx"):
            assert s.suffix_re.match(sfx) is None, sfx     # a solo suffix, the negative-control smoke and a mis-ordered tail are outside the shared grammar
        assert B.repeat_of_suffix(s, "bcbw-gemma4e4b_s1-8ba0b48-shared-rep2") == 2 and B.repeat_of_suffix(s, "bcbw-gemma4e4b_s1-8ba0b48-shared") is None
        assert B.suffix_task_for(s, "bcbw-gemma4e4b_s1-8ba0b48-shared-bcb") == "bigcodebench" and B.suffix_task_for(s, "bcbw-gemma4e4b_s1-8ba0b48-shared") is None
        assert s.shared.relaunch_entry_tag("r2") == "shared-r2"
    for solo in (BCF, BFF):
        assert solo.SPEC.shared is None and solo.SPEC.suffix_re is B.SUFFIX_RE and solo.SPEC.spelling_of("gemma4e4b_s1") == "gemma4e4b_s1" and solo.SPEC.schedule_file == B.SCHEDULE_FILE
        assert B.repeat_of_suffix(solo.SPEC, "bcbw-gemma4e4b_s1-8ba0b48-shared-rep2") is None and B.suffix_task_for(solo.SPEC, "bcbw-gemma4e4b_s1-8ba0b48-bcb") == "bigcodebench"
    assert B.row_okey("none", 2) == "none@rep2" and B.row_okey("none", None) == "none" and B.split_row_okey("none@rep2") == ("none", 2) and B.split_row_okey(RELU) == (RELU, None)
    assert B.COMPARISON_KINDS == ("shared_vs_solo", "repeat_floor", "it_none_comparison") and B.ANCHOR_KIND_BORROWED == "borrowed" and B.ANCHOR_KIND_SOLO_STAND_IN == "solo_stand_in" and B.REPEAT_CLASS == "repeat" and B.IT_NONE_CLASS == "unsteered"
    assert len(B.ROW_ENGINE_KEYS) == 8 and set(B.ROW_ENGINE_KEYS) == {"row_engine", "standup_id", "engine_cells", "max_loras", "residual_slot_armed", "compile_cache", "row_dir", "verification_pass"}


def test_the_spellings_resolve_to_the_solo_spellings_models_and_transfer_kinds(seeds):
    for solo_tag, shared_tag in (("gemma4e4b_s1", "gemma4e4b_s1_shared"), ("qwen35_4b_s1", "qwen35_4b_s1_shared"), ("llama8b_s1", "llama8b_s1_shared"), ("nemotron3nano4b_s1", "nemotron3nano4b_s1_shared"),
                                 ("gemma4e4b_it", "gemma4e4b_it_shared"), ("qwen35_4b_it", "qwen35_4b_it_shared"), ("llama8b_it", "llama8b_it_shared")):
        a, b = seeds.model_for_view_legacy(solo_tag), seeds.model_for_view_legacy(shared_tag)
        assert b["model_id"] == a["model_id"] and b["fit_model_id"] == a["fit_model_id"] and b["transfer_kind"] == a["transfer_kind"], (solo_tag, a, b)
        assert b["source"] == "legacy_spellings.yaml model_keys"
    assert seeds.model_for_view_legacy("gemma4e4b_s1p_shared")["model_id"] is None, "no paired row runs on the shared harness; no spelling is invented for it"


# --------------------------------------------------------------------------------------------
# the read
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("eval_id", ["bigcodebench", "bfcl"])
def test_every_store_files_under_the_shared_spelling_with_the_in_run_and_the_borrowed_contrasts(tmp_path, seeds, eval_id):
    fx, ad, bt, facts = _read(tmp_path, seeds, eval_id, repeat=True, it_none=True, relaunched=True)
    mod, solo = SH.SHARED[eval_id], SH.SOLO[eval_id]
    runs = SH.runs_by_key(bt)
    assert facts["runs"] == 5 and facts["n_refusals"] == 0 and facts["n_not_run"] == 0 and facts["n_parked"] == 1, facts["refusals"]
    assert set(runs) == {NONE, "none@rep2", RELU, PROMPTED, NONE + "|it"} or set(runs) == {NONE, "none@rep2", RELU, PROMPTED}, sorted(runs)
    by_view = {}
    for r in bt["run"]:
        by_view.setdefault(r["view_legacy"], []).append(r)
    assert set(by_view) == {"gemma4e4b_s1_shared", "gemma4e4b_it_shared"} and len(by_view["gemma4e4b_it_shared"]) == 1 and len(by_view["gemma4e4b_s1_shared"]) == 4
    inst = fx["instruments"][eval_id]
    for r in by_view["gemma4e4b_s1_shared"]:
        e, okey = r["extra"], r["campaign_cell_id_legacy"]
        rep = 2 if e["shared_standup"]["repeat"] else None
        assert r["producer_run_key_legacy"] == f"gemma4e4b_s1_shared/{fx['keys'][okey]}" + ("@rep2" if rep else "") and r["lane_source_legacy"] == mod.SOURCE
        assert r["run_id"] == mod.run_id_for(TAG, fx["keys"][okey], okey, inst, repeat=rep) == I.run_id(LC.CAMPAIGN, r["producer_run_key_legacy"], okey, eval_id, inst)
        assert e["comparison_kind"] == ("repeat_floor" if rep else "shared_vs_solo") and e["solo_run_id"] == fx["solo_run_ids"][okey][eval_id] == solo.run_id_for(TAG, fx["solo_keys"][okey], okey, inst)
        assert e["solo_run_id"] != r["run_id"] and e["solo_store"]["mirror_rel"] == B.rel_store(SH.ROW_DIR, okey, fx["solo_suffix"]) and e["solo_store"]["store_key"] == fx["solo_keys"][okey]
        assert e["standup_id"] == fx["standups"][okey] and e["row_engine"]["standup_id"] == e["standup_id"] and okey in e["row_engine"]["engine_object_keys"]
        assert e["row_engine"]["result_standup"] == e["standup_id"] and e["standup"]["standup_attempt_id"] == e["standup_id"] and e["gate"]["standup_id"] == e["standup_id"]
        assert e["comparison"]["state"] == "contrasted" and e["comparison"]["contrast_anchor_kind"] == "solo_stand_in" and e["comparison"]["value_solo"] == e["comparison"]["value_shared"]
        assert e["launch_set"]["class"] == ("repeat" if rep else {NONE: "anchor", RELU: "arm", PROMPTED: "prompted"}[okey]) and e["launch_set"]["spelling"] == "gemma4e4b_s1_shared" and e["launch_set"]["selection"] == "s1"
        assert e["gate"]["max_loras"] == e["row_engine"]["max_loras"] and e["gate"]["residual_slot_armed"] == e["row_engine"]["residual_slot_armed"] and e["row_engine"]["flags_agree"]
        assert e["gate"]["comparability_class"] == ("cross_standup" if okey in (RELU, NONE) and not rep else "same_standup")   # the none against the SOLO record; the relaunch on its own standup
        assert e["gate"]["comparison"]["comparison_kind"] == e["comparison_kind"] and e["row_engine"]["object_artifact_sha256_scalar"] is None
        assert e["producer_view_legacy"] == "s1" and e["seed_replicate_label"] == "s1" and r["trial_index"] == 0 and r["seed_base"] is None
        assert e["standup"]["served_object_keys"] == e["row_engine"]["engine_object_keys"] and e["shared_standup"]["solo_source"] == solo.SOURCE
    relu = runs[RELU]
    assert relu["extra"]["standup_id"] == SH.STANDUP_B != runs[NONE]["extra"]["standup_id"] == SH.STANDUP_A and relu["extra"]["launch_set"]["relaunch"]["entry_tag"] == "shared-r2"
    assert relu["extra"]["launch_set"]["relaunch"]["tag"] == "r2" and relu["extra"]["launch_set"]["recovery"]["class"] == "cross_standup"
    # the contrasts: the in-run reference (the row's shared none) and the borrowed one (the solo run) per row; the repeat both; the it none none
    in_run = {c["run_id"]: c for c in _contrasts(bt) if c["anchor_kind"] != "solo_stand_in"}
    borrowed = {c["run_id"]: c for c in _contrasts(bt, "solo_stand_in")}
    assert set(in_run) == {runs[k]["run_id"] for k in (RELU, PROMPTED, "none@rep2")} and set(borrowed) == {runs[k]["run_id"] for k in (NONE, RELU, PROMPTED, "none@rep2")}
    assert all(c["anchor_run_id"] == runs[NONE]["run_id"] for c in in_run.values())
    assert in_run[relu["run_id"]]["anchor_kind"] == "sibling_none" and in_run[relu["run_id"]]["extra"]["cross_standup"] is True and "standup" in in_run[relu["run_id"]]["extra"]["cross_standup_note"]
    assert in_run[runs[PROMPTED]["run_id"]]["anchor_kind"] == "same_standup_none" and in_run[runs["none@rep2"]["run_id"]]["anchor_kind"] == "same_standup_none"
    assert in_run[runs["none@rep2"]["run_id"]]["delta"] == 0.0 and in_run[runs["none@rep2"]["run_id"]]["extra"]["comparison_kind"] == "repeat_floor"
    for k, c in borrowed.items():
        r = next(x for x in bt["run"] if x["run_id"] == k)
        assert c["anchor_run_id"] == r["extra"]["solo_run_id"] and c["computed_by"] == "registry" and c["delta"] == 0.0 and c["delta_p"] == 1.0 and c["n_pairs"] >= 4
        e = c["extra"]
        assert e["value_shared"] == e["value_solo"] == r["extra"]["comparison"]["value_shared"] and e["cross_standup"] is True and e["standup_ids"] == {"row": r["extra"]["standup_id"], "anchor": None}
        assert e["solo_record"].endswith(f"cells/{fx['solo_keys'][r['campaign_cell_id_legacy']]}/outcomes/{mod.OUTCOMES_FILE}") and e["anchor_rule"] == B.SHARED_COMPARISON_RULE
        assert e["comparison_kind"] == r["extra"]["comparison_kind"] and e["producer_delta_value_vs_solo"] == 0.0
    assert facts["comparisons"][f"{TAG}|{RELU}"]["state"] == "contrasted" and facts["comparisons"][f"{TAG}|none@rep2"]["anchor_kind"] == "solo_stand_in"
    assert set(facts["standups"]) == {SH.STANDUP_A, SH.STANDUP_B, SH.STANDUP_IT} and facts["standups"][SH.STANDUP_A]["rows"] == [f"{TAG}|{NONE}", f"{TAG}|none@rep2", f"{TAG}|{PROMPTED}"]
    assert facts["parked"][0]["store"] == B.rel_store(SH.ROW_DIR, RELU, fx["solo_suffix"] + "-shared") and facts["parked"][0]["relaunched_to"].endswith("-shared-r2")
    assert facts["shared_source_rule"] == B.SHARED_SOURCE_RULE and facts["solo_source"] == solo.SOURCE and facts["solo_root_present"] is True
    # the source states name the shared launch rows by tuple id, the repeat and the it none included
    states = ad.source_states(X.ctx(), ad.discover())
    ls = next(s for s in states if s.input_uri.endswith(f"launch_sets_shared.json#{GROUP}"))
    assert sorted(ls.current_members) == sorted(["s1/none", "s1/none@rep2", f"s1/{RELU}", f"s1/{PROMPTED}", "it/none"])


def test_the_instruct_rows_own_none_files_as_a_comparison_row_that_anchors_nothing_and_contrasts_nothing(tmp_path, seeds):
    fx, ad, bt, facts = _read(tmp_path, seeds, it_none=True, repeat=False)
    it = next(r for r in bt["run"] if r["view_legacy"] == "gemma4e4b_it_shared")
    e = it["extra"]
    assert it["campaign_cell_id_legacy"] == NONE and it["object_id"] == "none" and it["transfer_kind"] == "none" and it["model_id"] == "gemma4e4b_it" and it["seed_base"] is None
    assert it["producer_run_key_legacy"] == f"gemma4e4b_it_shared/{fx['it_key']}" and it["run_id"] == BCS.run_id_for(IT_TAG, fx["it_key"], NONE, fx["instruments"]["bigcodebench"])
    assert e["comparison_kind"] == "it_none_comparison" and e["solo_run_id"] is None and e["solo_store"] is None and e["comparison"]["state"] == "no_contrast" and e["comparison"]["contrast_anchor_kind"] is None
    assert e["launch_set"]["class"] == "unsteered" and e["launch_set"]["selection"] == "it" and e["instruct_direct"]["own_none"] is True and e["instruct_direct"]["transfer_kind_recorded"] == "none"
    assert "anchor_alias" not in e and "anchor" not in e and e["standup_id"] == SH.STANDUP_IT
    assert it["run_id"] not in {c["run_id"] for c in bt["run_contrast"]} and it["run_id"] not in {c["anchor_run_id"] for c in bt["run_contrast"]}
    assert facts["comparisons"][f"{IT_TAG}|{NONE}"]["state"] == "no_contrast" and "anchor_aliases" not in facts and facts["not_run"] == []
    # hazard 5: its lane_roster cites the roster of ANOTHER tree -- matched by hash within the set, the tree recorded
    assert e["launched_roster"]["roster_file"] == f"{Z.ROSTER_FILE[:-5]}.{Z.TREE2[:7]}.yaml" and e["launched_roster"]["tree_sha"] == Z.TREE2 and e["served_tree_sha"] == Z.TREE
    # hazard 4: lane_roster.json:wave is the LAUNCH tag, never the `_shared` spelling
    assert e["launched_roster"]["launch_tag_legacy"] == IT_TAG and e["gate"]["comparability_note"] and e["gate"]["comparability_class"] is None


# --------------------------------------------------------------------------------------------
# the eight identity hazards (REQUIRED_RECORDS.md section 6)
# --------------------------------------------------------------------------------------------

def test_hazard_1_run_ids_are_minted_under_the_shared_spelling_and_a_solo_id_that_is_not_the_solo_readers_refuses(tmp_path, seeds):
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=True)
    runs = SH.runs_by_key(bt)
    inst = fx["instruments"]["bigcodebench"]
    # the SOLO reader over the solo root beside it mints exactly the ids the shared rows name as solo_run_id -- and none of the shared ids
    solo_bt, _f = Z.read(SH.solo_adapter("bigcodebench", fx, seeds))
    solo_ids = {r["campaign_cell_id_legacy"]: r["run_id"] for r in solo_bt["run"] if r["view_legacy"] == TAG}
    for okey in (NONE, RELU, PROMPTED):
        assert runs[okey]["extra"]["solo_run_id"] == solo_ids[okey] == BCS.solo_run_id_for(TAG, fx["solo_keys"][okey], okey, inst)
        assert runs[okey]["run_id"] != solo_ids[okey] and runs[okey]["producer_run_key_legacy"].startswith("gemma4e4b_s1_shared/")
        # the same store key on both sides (K = 1): the tag prefix alone tells the ids apart
        assert fx["keys"][okey] == fx["solo_keys"][okey] and I.run_id(LC.CAMPAIGN, f"{TAG}/{fx['keys'][okey]}", okey, "bigcodebench", inst) == solo_ids[okey]
    assert runs["none@rep2"]["run_id"] not in set(solo_ids.values()) | {runs[NONE]["run_id"]} and runs["none@rep2"]["producer_run_key_legacy"].endswith("@rep2")
    # a launch row naming a solo_run_id the solo reader does not mint refuses by name (the siblings file)
    fx2, ad2, bt2, facts2 = _read(tmp_path / "b", seeds, repeat=True, solo_run_id_override="sdfbase_full22:bigcodebench:461f34836b51:0000000000000000")
    why = _refusal(facts2, f"{TAG}|{NONE}")
    assert "solo_run_id" in why and "the solo reader mints" in why and facts2["runs"] == 2, facts2["refusals"]     # the RELU and the prompted key file; the repeat names the same wrong id
    assert set(SH.runs_by_key(bt2)) == {RELU, PROMPTED} and "the solo reader mints" in _refusal(facts2, f"{TAG}|none@rep2")


def test_hazard_2_the_repeat_row_is_keyed_by_its_segment_and_two_rows_on_one_key_refuse(tmp_path, seeds):
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=True)
    runs = SH.runs_by_key(bt)
    assert runs[NONE]["run_id"] != runs["none@rep2"]["run_id"] and runs[NONE]["object_id"] == runs["none@rep2"]["object_id"] == "none"
    assert runs["none@rep2"]["extra"]["launch_set"]["tuple_id"] == "s1/none@rep2" and runs["none@rep2"]["extra"]["launch_set"]["repeat"]["key"] == "none@rep2"
    assert runs["none@rep2"]["extra"]["shared_standup"]["repeat"] == 2 and runs["none@rep2"]["extra"]["launch_set"]["class"] == "repeat"
    assert facts["rows"][f"{TAG}|none@rep2"]["repeat"] == 2 and facts["rows"][f"{TAG}|{NONE}"]["repeat"] is None
    tn = runs["none@rep2"]["extra"]["three_numbers"]
    assert tn["producer_record_agrees"] is True
    # the repeat row spelled WITHOUT its segment on launched_as.suffix collides with the none's key: both rows refuse by name and the repeat store binds nothing
    fx2, ad2, bt2, facts2 = _read(tmp_path / "b", seeds, repeat=True, repeat_row_without_segment=True)
    assert facts2["runs"] == 2 and set(SH.runs_by_key(bt2)) == {RELU, PROMPTED}
    why = _refusal(facts2, f"{TAG}|{NONE}")
    assert "names 2 rows for s1/none" in why and B.SHARED_ROW_KEY_RULE[:40] in why
    assert "names no row for s1/none@rep2" in _refusal(facts2, f"{TAG}|none@rep2")


def test_hazard_3_the_solo_spelling_under_view_refuses_by_name(tmp_path, seeds):
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=False, spelling_override=TAG)
    why = _refusal(facts, f"{TAG}|{NONE}")
    assert f"spells view {TAG!r}" in why and "gemma4e4b_s1_shared" in why and facts["runs"] == 2
    # the three-number record's `view` is the registry spelling too
    fx2, ad2, bt2, facts2 = _read(tmp_path / "b", seeds, repeat=False, three_override=lambda t: t["rows"][f"{TAG}/{NONE}"].__setitem__("view", "s1"))
    assert "view: producer 's1', re-derived 'gemma4e4b_s1_shared'" in _refusal(facts2, f"{TAG}|{NONE}")


def test_hazard_4_lane_roster_wave_is_the_launch_tag_never_the_shared_spelling(tmp_path, seeds):
    fx = SH.mirror(tmp_path, seeds, eval_id="bigcodebench", repeat=False)
    p = os.path.join(fx["root"], B.STORES_HEAD, SH.ROW_DIR, NONE, B.THINKING_DIR, fx["solo_suffix"] + "-shared", B.LAUNCHED_ENTRY_FILE)
    lr = json.load(open(p, encoding="utf-8"))
    lr["wave"] = "gemma4e4b_s1_shared"
    X.write(p, lr)
    out, _ = SH.read_all(SH.adapter("bigcodebench", fx, seeds))
    bt, facts = out[GROUP]
    assert "names launch tag 'gemma4e4b_s1_shared', not 'gemma4e4b_s1'" in _refusal(facts, f"{TAG}|{NONE}") and facts["runs"] == 2


def test_hazard_7_the_parked_first_attempt_mirrored_complete_parks_and_never_reads_or_refuses(tmp_path, seeds):
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=False, relaunched=True)
    runs = SH.runs_by_key(bt)
    assert facts["runs"] == 3 and facts["n_refusals"] == 0 and facts["n_parked"] == 1
    parked = facts["parked"][0]
    assert parked["suffix"] == fx["solo_suffix"] + "-shared" and parked["object_key"] == RELU and parked["relaunched_to"] == fx["solo_suffix"] + "-shared-r2" and parked["parked_suffixes"] == [parked["suffix"]]
    assert runs[RELU]["extra"]["store_suffix_legacy"].endswith("-shared-r2") and runs[RELU]["extra"]["launch_set"]["relaunched_from"]["suffix"] == parked["suffix"]
    assert os.path.isfile(os.path.join(fx["root"], B.STORES_HEAD, SH.ROW_DIR, RELU, B.THINKING_DIR, parked["suffix"], "RESULT")), "the parked store IS complete and mirrored"
    assert runs[RELU]["extra"]["launch_set"]["relaunch"]["entry_tag"] == "shared-r2" and runs[RELU]["extra"]["launch_set"]["relaunch"]["standup_id"] == SH.STANDUP_B
    # the relaunched cell's in-run reference is the row's none served by ANOTHER standup: sibling_none, the gate's cross_standup
    c = next(c for c in _contrasts(bt) if c["run_id"] == runs[RELU]["run_id"] and c["anchor_kind"] != "solo_stand_in")
    assert c["anchor_kind"] == "sibling_none" and c["extra"]["standup_ids"] == {"row": SH.STANDUP_B, "anchor": SH.STANDUP_A} and runs[RELU]["extra"]["anchor"]["cross_standup"] is True
    assert runs[RELU]["extra"]["gate"]["comparability_class"] == "cross_standup" and runs[RELU]["extra"]["gate"]["compared_against_standup_ids"] == {"this": SH.STANDUP_B, "prior": SH.STANDUP_A}


def test_the_standup_identity_and_the_row_engine_block_refuse_by_name(tmp_path, seeds):
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=False, standup_override=hashlib.md5(b"another standup").hexdigest())
    assert "RESULT's last line carries standup=" in _refusal(facts, f"{TAG}|{NONE}") and B.SHARED_STANDUP_RULE[:30] in _refusal(facts, f"{TAG}|{NONE}") and facts["runs"] == 2
    fx2, ad2, bt2, facts2 = _read(tmp_path / "b", seeds, repeat=False, row_engine_keys_override=lambda blk: blk.pop("compile_cache"))
    assert "row_engine carries the keys" in _refusal(facts2, f"{TAG}|{NONE}") and "not exactly" in _refusal(facts2, f"{TAG}|{NONE}")
    # one store of the row launch spelling another standup id: the LAUNCH-level check fires first (one engine bring-up serves a row launch) and refuses every store of it
    fx3, ad3, bt3, facts3 = _read(tmp_path / "c", seeds, repeat=False, row_engine_keys_override=lambda blk: blk.__setitem__("standup_id", hashlib.md5(b"x").hexdigest()))
    assert facts3["runs"] == 0 and all("spell 2 standup ids" in x for x in facts3["refusals"]) and len(facts3["refusals"]) == 3, facts3["refusals"]
    assert len(facts3["standup_launches"][f"{SH.ROW_DIR}/{fx3['solo_suffix']}-shared"]) == 2
    fx4, ad4, bt4, facts4 = _read(tmp_path / "d", seeds, repeat=False, gate_class_override="cross_standup")     # the ReLU key's gate names the row's none on the SAME standup
    assert "comparability_class 'cross_standup' disagrees with its standup_ids" in _refusal(facts4, f"{TAG}|{RELU}") and facts4["runs"] == 2


def test_a_solo_mirror_without_the_store_leaves_the_run_filed_with_the_comparison_waiting(tmp_path, seeds):
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=True, no_solo_stores=True)
    runs = SH.runs_by_key(bt)
    assert facts["runs"] == 4 and facts["n_refusals"] == 0 and not _contrasts(bt, "solo_stand_in")
    assert {c["anchor_kind"] for c in _contrasts(bt)} == {"same_standup_none"} and len(_contrasts(bt)) == 3, "the in-run contrasts stand; the borrowed ones wait"
    for k, r in runs.items():
        assert r["extra"]["comparison"]["state"] == "solo_store_absent" and "not mirrored under the solo root" in r["extra"]["comparison"]["note"] and r["extra"]["solo_run_id"]
        assert facts["comparisons"][f"{TAG}|{k}"]["state"] == "solo_store_absent"
    assert facts["solo_root_present"] is True and not any(os.path.relpath(i.uri.rsplit("#", 1)[0], fx["root"]).startswith("..") for i in ad.discover()), "no solo-side input when no solo store is mirrored"


def test_a_solo_run_that_is_not_current_on_the_frame_leaves_the_borrowed_comparison_waiting_by_name(tmp_path, seeds):
    """Emission 3: the solo run stands in as the anchor of the borrowed comparison and is never re-picked; its CURRENCY is checked against the pass's
    frame. On the steady-state frame (the solo runs current) every borrowed comparison files with `extra.anchor_selection` (currency only); with the
    solo `none` retired (an exclude ruling on the solo source) that one comparison waits -- `extra.comparison.state` solo_not_current, a refusal line,
    the park counted -- while the other borrowed comparisons and every in-run contrast file; a frame that never folded the solo runs parks them all
    (`not_in_frame`: the first pass before the solo source's fold)."""
    fx = SH.mirror(tmp_path, seeds, eval_id="bigcodebench", repeat=False)
    ad = SH.adapter("bigcodebench", fx, seeds)
    out, _ = SH.read_all(ad)
    bt, facts = out[GROUP]
    borrowed = _contrasts(bt, "solo_stand_in")
    assert len(borrowed) == 3 and all(c["extra"]["anchor_selection"]["state"] == "own_current" and c["extra"]["anchor_selection"]["anchor_kind"] == "solo_stand_in" for c in borrowed)
    assert all("currency_only" in c["extra"]["anchor_selection"] and c["extra"]["currency_rule"] == B.CURRENCY_RULE for c in borrowed)
    assert facts["anchor_states"] == {"own_current": 5} and facts["contrasts_filed"] == 5 and facts["contrasts_parked"] == 0, facts["anchor_states"]
    # the solo none retired on the frame: its borrowed comparison waits by name; the two others and the in-run contrasts stand
    frame = X.primed_frame(ad)
    solo_none = fx["solo_run_ids"][NONE]["bigcodebench"]
    AF.retire(frame, solo_none, "selection_state excluded (decisive ruling exclude r_synth_exclude_solo_none)")
    out2, _ = SH.read_all(ad, conn=frame.con)
    bt2, facts2 = out2[GROUP]
    runs = SH.runs_by_key(bt2)
    borrowed2 = {c["run_id"]: c for c in _contrasts(bt2, "solo_stand_in")}
    assert set(borrowed2) == {runs[RELU]["run_id"], runs[PROMPTED]["run_id"]} and len(_contrasts(bt2)) == 4
    none_run = runs[NONE]
    assert none_run["extra"]["comparison"]["state"] == B.SOLO_STATE_NOT_CURRENT == "solo_not_current" and solo_none in none_run["extra"]["comparison"]["note"]
    assert none_run["extra"]["comparison"]["anchor_selection"]["own_reason"] == "excluded" and facts2["comparisons"][f"{TAG}|{NONE}"]["state"] == "solo_not_current"
    assert facts2["contrasts_parked"] == 1 and facts2["anchor_states"] == {"own_current": 4, "no_current_anchor": 1} and facts2["n_refusals"] == 1
    assert _refusal(facts2, f"{TAG}|{NONE}|solo_contrast").startswith(f"{TAG}|{NONE}|solo_contrast -- contrast has no current anchor: the solo run {solo_none} is not a current run")
    # a frame that never folded the solo runs (the shared source read before the solo source's fold): every borrowed comparison waits, not_in_frame
    frame3 = AF.Frame()
    AF.load_records(frame3, AF.priming_read(ad, X.ctx(frame3.con)))
    out3, _ = SH.read_all(ad, conn=frame3.con)
    bt3, facts3 = out3[GROUP]
    assert not _contrasts(bt3, "solo_stand_in") and len(_contrasts(bt3)) == 2 and facts3["contrasts_parked"] == 3
    assert {r["extra"]["comparison"]["state"] for r in bt3["run"]} == {"solo_not_current"} and {facts3["comparisons"][k]["currency_own_reason"] for k in facts3["comparisons"]} == {"not_in_frame"}


def test_the_solo_stores_are_the_groups_inputs_so_a_re_mirrored_solo_store_re_reads_the_comparison(tmp_path, seeds):
    fx = SH.mirror(tmp_path, seeds, eval_id="bigcodebench", repeat=False)
    ad = SH.adapter("bigcodebench", fx, seeds)
    inputs = ad.discover()
    solo_side = [i for i in inputs if os.path.relpath(i.uri.rsplit("#", 1)[0], fx["root"]).startswith("..")]
    assert len(solo_side) == 6 and all(i.uri.endswith(f"#{GROUP}") for i in solo_side) and all(ad.comparison_group(i) == GROUP for i in solo_side)
    names = sorted(os.path.basename(i.uri.rsplit("#", 1)[0]) for i in solo_side)
    assert names == sorted(["RESULT"] * 3 + [BCF.OUTCOMES_FILE] * 3), names        # the solo layout carries no top-level manifest.json today
    assert all(i.uri.rsplit("#", 1)[0].startswith(fx["solo_root"]) for i in solo_side)


def test_a_solo_side_alias_carries_its_group_in_the_identity_on_the_grading_host_path_and_under_a_map(tmp_path, seeds):
    """The solo-side aliases go through `Input.with_alias` (P3b's rule: the identity FOLLOWS the alias). On the grading host path the identity IS the aliased
    uri -- the inputs_json / watermark / revision-id key, one per input; under an identity map the canonical spelling carries the alias. A
    `dataclasses.replace(inp, uri=...)` left every solo-side alias with the UNSUFFIXED identity (the gate's guard on 4ef0b04c0)."""
    fx = SH.mirror(tmp_path, seeds, eval_id="bigcodebench", repeat=False)
    with AB.using_identity_map(None):
        inputs = SH.adapter("bigcodebench", fx, seeds).discover()
    solo_side = [i for i in inputs if os.path.relpath(i.uri.rsplit("#", 1)[0], fx["root"]).startswith("..")]
    assert len(solo_side) == 6 and all(i.identity_uri == i.uri and i.uri.endswith(f"#{GROUP}") for i in solo_side), [(i.uri, i.identity_uri) for i in solo_side]
    assert len({i.identity_uri for i in inputs}) == len(inputs) == len({i.uri for i in inputs})
    canon_shared, canon_solo = "/home/someone-else/canonical/shared", "/home/someone-else/canonical/solo"
    with AB.using_identity_map(AB.IdentityMap.build({fx["root"]: canon_shared, fx["solo_root"]: canon_solo})):
        inputs_m = SH.adapter("bigcodebench", fx, seeds).discover()
    by_uri = {i.uri: i for i in inputs_m}
    assert sorted(by_uri) == sorted(i.uri for i in inputs), "the map respells identities, never what the adapter discovers"
    for i in solo_side:
        assert by_uri[i.uri].identity_uri == canon_solo + i.uri[len(fx["solo_root"]):] and by_uri[i.uri].identity_uri.endswith(f"#{GROUP}"), (i.uri, by_uri[i.uri].identity_uri)
    assert len({i.identity_uri for i in inputs_m}) == len(inputs_m)


# --------------------------------------------------------------------------------------------
# hazard 8: the frame -- what the roster and the rulings do with a shared run
# --------------------------------------------------------------------------------------------

def _registry_root(tmp_path, seeds, objects):
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    # (no stand-in paper_round: the synthetic roster's round is the ONE open round, so v_roster_match reads its entries)
    recs = DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in objects:
        r = LC.standin_object_record(o, seeds, F.producer(), now, "gemma4e4b_base")
        if r:
            recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def _roster_over_the_full_instrument(round_id: str = "r_shared_synth") -> dict:
    """A mini roster (the roster test's shape) naming gemma4e4b_it x bigcodebench x {none, constitution_relu} on the FULL item set -- the way the appendix
    transfer comparison names the full-suite rows; no spelling, no source: v_roster_match knows neither."""
    return {"meta": {"schema": RO.ROSTER_SCHEMA, "roster_version": "v1", "round_id": round_id,
                     "paper_round": {"round_id": round_id, "opened_utc": "2026-09-01", "description": "synthetic round over the full item set", "paper_build_commit": "e" * 40}},
            "vocabulary": {"eval_id_from_eval_legacy": {}},
            "models": {"gemma4e4b_it": {"model_legacy": "gemma4e4b", "paper_lane": {"transfer_kind": "base_to_instruct"}}},
            "objects": {"none": {"paper_label": "none", "object_kind": "none", "object_role": "baseline_none"},
                        "constitution_relu": {"paper_label": "constitution_relu", "object_kind": "relu_steer", "object_role": "constitution", "arm_legacy": "const_psr"}},
            "judges": {},
            "evals": {"bigcodebench": {"eval_category": "capability", "component_legacy": "capability", "judge": None,
                                       "instrument_selector_json": {"harness": "inspect_evals", "item_set": BCF.ITEM_SET_ID}}},
            "comparisons": [{"comparison_id_template": "fig_synth_full/{transfer_kind}", "figure_legacy": "fig_synth_full",
                             "models": [{"model_id": "gemma4e4b_it", "transfer_kind": "base_to_instruct"}],
                             "objects": ["none", "constitution_relu"], "evals": ["bigcodebench"], "strength_rules": ["operating_point"],
                             "source_citation": "synthetic"}]}


def _ruling(**over):
    d = {"ruling_id": "R_x", "round_id": "r_shared_synth", "action": "exclude", "scope_kind": "key", "scope_json": {}, "target_json": {}, "reason": "synthetic",
         "ruled_utc": "2026-09-28T00:00:00Z", "ruled_by": "the registry's owner", "source_ref": "test", "provenance": {"kind": "user_direct"}}
    d.update(over)
    return d


def _load_roster(root: str, tmp_path, roster: dict, rulings=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ry = tmp_path / "roster.yaml"
    ry.write_text(yaml.safe_dump(roster, sort_keys=False))
    rj = tmp_path / "rulings.jsonl"
    rj.write_text("".join(json.dumps(r) + "\n" for r in rulings))
    load = RO.load(str(ry), str(rj) if rulings else None)
    RO.write_records(root, load, written_utc=synth.t(50))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return load


def test_hazard_8_a_shared_run_a_roster_entry_matches_reads_selected_until_an_exclude_covers_the_shared_sources(tmp_path, seeds):
    """The claim "no roster selector names the `_shared` spelling, so every shared row reads not_in_roster by construction" (REQUIRED_RECORDS.md sections 0 / 6.8)
    holds ONLY for rows no entry matches: `v_roster_match` matches on model / eval / object / instrument -- never on view_legacy or the source -- so a shared run
    whose solo twin an entry matches is matched too. Ingested beside its solo twin, with the solo source held `selected` by an include ruling (the live
    `r2v4-include-full-suite-...` on the two solo sources), the shared run reads `selected` beside it (two runs at one key in the frame) and only an `exclude`
    scoped `match.lane_sources_legacy [<the two shared sources>]` takes it out; the include precedes the `later` rule, so the newer shared run never supersedes the
    solo. The remedy is a ruling for the registry's owner to file before the first real mirror, never a reader default."""
    fx = SH.mirror(tmp_path, seeds, eval_id="bigcodebench", repeat=False)
    s2 = Z.seeds_for(seeds)
    solo_ad = SH.solo_adapter("bigcodebench", fx, seeds)
    shared_ad = SH.adapter("bigcodebench", fx, seeds)
    objects = [solo_ad.resolve_object(RELU, "s1", pin_key=Z.STAGED, model_key=SH.MK)[0], solo_ad.resolve_object(Z.COMPOSED, "s1p", pin_key=Z.STAGED_P, model_key=SH.MK)[0]]
    root = _registry_root(tmp_path, s2, objects)
    rep, ing, ing2 = F.run_and_ingest(root, solo_ad, now="2026-09-28T01:00:00.000Z")
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rep2, ing3, ing4 = F.run_and_ingest(root, shared_ad, now="2026-09-28T02:00:00.000Z")
    assert rep2.groups_refused == 0 and ing3.refused + ing4.refused == 0, ([r.error for r in rep2.results], ing3.refusals, ing4.refusals)
    solo_none = BCF.run_id_for(TAG, fx["solo_keys"][NONE], NONE, fx["instruments"]["bigcodebench"])
    shared_none = BCS.run_id_for(TAG, fx["keys"][NONE], NONE, fx["instruments"]["bigcodebench"])
    shared_relu = BCS.run_id_for(TAG, fx["keys"][RELU], RELU, fx["instruments"]["bigcodebench"])
    rows = F.q(root, "SELECT run_id, view_legacy, lane_source_legacy, lifecycle_state FROM run WHERE run_id IN (?, ?, ?)", (solo_none, shared_none, shared_relu))
    assert sorted(rows) == sorted([(solo_none, TAG, BCF.SOURCE, "ingested"), (shared_none, f"{TAG}_shared", BCS.SOURCE, "ingested"), (shared_relu, f"{TAG}_shared", BCS.SOURCE, "ingested")]), rows
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast WHERE anchor_kind = 'solo_stand_in' AND run_id IN (?, ?)", (shared_none, shared_relu)) == [(2,)], "the borrowed contrasts ingest against the solo runs"
    # (a) the roster on the full item set, no rulings: the entry matches the solo none AND the shared none / relu alike (no spelling, no source in the selector)
    _load_roster(root, tmp_path, _roster_over_the_full_instrument())
    matched = {r[0] for r in F.q(root, "SELECT run_id FROM v_roster_match WHERE run_id IN (?, ?, ?)", (solo_none, shared_none, shared_relu))}
    assert matched == {solo_none, shared_none, shared_relu}, matched
    state = {r[0]: (r[1], r[2]) for r in F.q(root, "SELECT run_id, selection_state, in_current_paper FROM v_paper_current WHERE run_id IN (?, ?, ?)", (solo_none, shared_none, shared_relu))}
    assert state[shared_none] == ("selected", 1) and state[shared_relu] == ("selected", 1), state      # THE HAZARD: a matched shared run enters the frame beside its twin
    assert state[solo_none] == ("selected", 1)
    # (b) the live shape: an include on the SOLO sources (the r2v4 include's scope) and an exclude on the SHARED sources -- the solo stays selected (the include
    #     precedes the later rule even though the shared run is newer), the shared rows read excluded and leave the frame
    rulings = [_ruling(ruling_id="R_include_solo", action="include", scope_json={"match": {"lane_sources_legacy": [BCF.SOURCE, BFF.SOURCE], "item_set_ids": [BCF.ITEM_SET_ID, BFF.ITEM_SET_ID]}},
                       ruled_utc="2026-09-27T00:00:00Z"),
               _ruling(ruling_id="R_exclude_shared", action="exclude", scope_json={"match": {"lane_sources_legacy": [BCS.SOURCE, BFS.SOURCE]}}, ruled_utc="2026-09-28T00:00:00Z",
                       reason="D583: a comparison source, never in the paper frame")]
    _load_roster(root, tmp_path / "r2", _roster_over_the_full_instrument("r_shared_synth"), rulings)
    state = {r[0]: (r[1], r[2], r[3]) for r in F.q(root, "SELECT run_id, selection_state, in_current_paper, decisive_ruling_id FROM v_paper_current WHERE run_id IN (?, ?, ?)", (solo_none, shared_none, shared_relu))}
    assert state[solo_none] == ("selected", 1, "R_include_solo") and state[shared_none] == ("excluded", 0, "R_exclude_shared") and state[shared_relu] == ("excluded", 0, "R_exclude_shared"), state
    # (c) a shared row NO entry matches (the prompted key: not in the roster's objects) is covered by the source-scoped exclude all the same:
    # since migration 0027 the decisive ruling reads by name whatever the roster says, so it reads excluded, never a paper row
    shared_prompted = BCS.run_id_for(TAG, fx["keys"][PROMPTED], PROMPTED, fx["instruments"]["bigcodebench"])
    assert F.q(root, "SELECT selection_state, in_current_paper FROM v_paper_current WHERE run_id = ?", (shared_prompted,)) == [("excluded", 0)]


# --------------------------------------------------------------------------------------------
# registration and the declared sources
# --------------------------------------------------------------------------------------------

def test_the_readers_are_registered_declared_and_versioned():
    from sisteer.registry import adapters as ADP
    reg = ADP.load_all()
    for mod, cls, solo in ((BCS, BCS.BigcodebenchFullSharedAdapter, BCF), (BFS, BFS.BfclFullSharedAdapter, BFF)):
        assert mod.NAME in reg and isinstance(reg[mod.NAME](), cls) and cls.covers_sources == (mod.SOURCE,) and AB.owner_of(cls()) == "Overflow" == mod.OWNER
        # version 2 since the store-attestation decision (2026-09-29): the live database holds accepted zero-run version-1 revisions of both sources, and an unchanged
        # group is re-read only on an input or emission change, so the corrected reader must publish under a new emission id to supersede them
        assert AB.emission_version_of(cls) == "4" and cls.SPEC.shared is not None and mod.SOLO_SOURCE == solo.SOURCE and mod.SOLO_ROOT_DEFAULT == solo.ROOT_DEFAULT   # 4: solo_stand_in (migration 0027)
        assert cls(root="/nonexistent/x").status_sources("gemma4e4b_s1") == (mod.SOURCE,) and cls(root="/nonexistent/x").solo_root == solo.ROOT_DEFAULT
        assert cls(root="/nonexistent/x", solo_root="/elsewhere").solo_root == "/elsewhere" and solo.BigcodebenchFullAdapter(root="/n").solo_root is None if solo is BCF else True
        assert mod.recount_computation_id() != solo.recount_computation_id() and mod.RECOUNT_ALGORITHM.endswith(f"{mod.NAME}.recount")
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "registry", "sources_r2_v1.yaml"), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = {r["source"]: r["owner"] for r in data["sources"]}
    adds = {a["source"]: a for a in data["meta"]["additions"]}
    for s in (BCS.SOURCE, BFS.SOURCE):
        assert names[s] == "the authors" and adds[s]["provisional"] is True and adds[s]["paper_build_commit"] is None and "COMPARISON" in adds[s]["why"]


# --------------------------------------------------------------------------------------------
# the real synthetic fixture (read-only; on the registry's machine)
# --------------------------------------------------------------------------------------------

@REAL_FIXTURE
@pytest.mark.parametrize("eval_id, n_items", [("bigcodebench", 1140), ("bfcl", 3981)])
def test_overflows_fixture_dry_reads_pass_on_both_readers(eval_id, n_items, seeds):
    """The dry read of 2026-09-28: 11 stores per root -> 10 runs, 8 in-run + 9 borrowed contrasts on the gemma row, the qwen instruct row's own
    none, one parked store, no refusal; the shared ids REQUIRED_RECORDS.md section 6.1 predicts; the solo ids the live solo readers mint (and the live registry holds).
    Since the rebuild of 2026-09-29 the fixture carries a 12th store, the qwen instruct row's own none served a second time (`-shared-rep2`): a repeat_floor row
    with its in-run floor against the row's none and no solo twin (SHARED_IT_REPEAT_RULE). Every endpoint of the fixture names its own store's key, so no store carries
    the landing-1 note (SHARED_ENDPOINT_TUPLE_RULE) and every `row_dir` is relative (SHARED_ROW_DIR_RULE: recorded, tail-bound)."""
    mod = SH.SHARED[eval_id]
    ad = SH.ADAPTERS[eval_id](root=os.path.join(FIXTURE_ROOT, mod.SOURCE), seeds=seeds)
    out, inputs = SH.read_all(ad)
    assert set(out) == {"gemma4e4b_s1", "qwen35_4b_s1"}
    bt, facts = out["gemma4e4b_s1"]
    assert facts["runs"] == 9 and facts["n_refusals"] == 0 and facts["n_parked"] == 1 and facts["n_not_run"] == 0, facts["refusals"]
    runs = SH.runs_by_key(bt)
    assert set(runs) == {"none", "none@rep2", "constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce",
                         "constv2-mix-full__lora_r4_ce_minus_ctl", "prompted__safety_refusal", "prompted__si_constitution"}
    assert {r["view_legacy"] for r in bt["run"]} == {"gemma4e4b_s1_shared"} and {r["extra"]["comparison_kind"] for r in bt["run"]} == {"shared_vs_solo", "repeat_floor"}
    inst = str(seeds.instrument_row(mod.INSTRUMENT_SEED, eval_id, model_id="gemma4e4b_it")["instrument_id"])
    none = runs["none"]
    assert none["run_id"] == I.run_id(LC.CAMPAIGN, "gemma4e4b_s1_shared/none__controls__t0__a0__2ff9cdff__a03d0582", "none", eval_id, inst)
    assert none["extra"]["solo_run_id"] == BCF.run_id_for("gemma4e4b_s1", "none__controls__t0__a0__2ff9cdff__a03d0582", "none", inst) if eval_id == "bigcodebench" else True
    if eval_id == "bigcodebench":
        assert none["run_id"].endswith(":6204f83a5e5957ef") and none["extra"]["solo_run_id"] == "sdfbase_full22:bigcodebench:461f34836b51:16e48a32f94a652e"      # section 6.1's two ids
    assert {r["extra"]["standup_id"] for r in bt["run"]} == {"d4bec50a106a17115204d9a0aec1339e", "c11d81a6da29690854e95c2bf0053abe"}
    assert runs["constv2ctl2-full__lora_r4_ce"]["extra"]["standup_id"] == "c11d81a6da29690854e95c2bf0053abe" and runs["constv2ctl2-full__lora_r4_ce"]["extra"]["gate"]["comparability_class"] == "cross_standup"
    assert {r["extra"]["served_tree_sha"][:7] for r in bt["run"]} == {"8ba0b48", "f1cd7f2"}, "hazard 6: two producer trees in one row, every row filed"
    kinds = sorted(c["anchor_kind"] for c in bt["run_contrast"])
    assert kinds == ["same_standup_none"] * 7 + ["sibling_none"] + ["solo_stand_in"] * 9, kinds
    for c in bt["run_contrast"]:
        assert c["n_pairs"] == n_items and c["computed_by"] == "registry"
        if c["anchor_kind"] == "solo_stand_in":
            assert c["delta"] == 0.0 and c["extra"]["value_shared"] == c["extra"]["value_solo"] and c["extra"]["producer_delta_value_vs_solo"] == 0.0
    assert facts["parked"][0]["store"] == "native/gemma4e4b-f22s1/constv2ctl2-full__lora_r4_ce/thinking_off/bcbw-gemma4e4b_s1-f1cd7f2-shared"
    assert facts["n_endpoint_served_key_mismatch"] == 0 and facts["endpoint_served_key_mismatch"] == []
    for r in bt["run"]:
        re_ = r["extra"]["row_engine"]
        assert re_["endpoint_served_key_agrees"] is True and re_["endpoint_served_key_note"] is None and re_["object_id_confidence"]["downgraded"] is False
        assert re_["object_id_confidence"]["filed"] == r["field_confidence_json"]["object_id"]["confidence"] and "landing 1" not in r["field_confidence_json"]["object_id"]["source"]
        # the row's container is the primary launch's (8ba0b48, hazard 6: the f1cd7f2 copies ride it, their tree recorded as disagreeing); the relaunch's is its own (f1cd7f2, -r2)
        relaunch = re_["standup_id"] == "c11d81a6da29690854e95c2bf0053abe"
        assert re_["row_record_dir_absolute"] is False and re_["row_record_dir"] == re_["row_record_dir_tail"] == ("rows/gemma4e4b-f22s1/bcbw-gemma4e4b_s1-f1cd7f2-shared-r2" if relaunch else "rows/gemma4e4b-f22s1/bcbw-gemma4e4b_s1-8ba0b48-shared")
        assert re_["row_record_dir_tree_agrees"] == (relaunch or r["extra"]["served_tree_sha"].startswith("8ba0b48")) and re_["row_record_dir_declared"] is None
        assert re_["served_map_source"] == B.SERVED_MAP_SOURCE_BOTH and re_["manifest_route_map_null"] is False
    bt_q, facts_q = out["qwen35_4b_s1"]
    assert facts_q["runs"] == 2 and facts_q["n_refusals"] == 0 and facts_q["n_endpoint_served_key_mismatch"] == 0, facts_q["refusals"]
    # 12 stores per root since the rebuild (2026-09-29 18:38Z): the gemma row's 9 runs + the parked first attempt, the qwen instruct row's none and its repeat
    stores = sorted(os.path.relpath(d, os.path.join(ad.root, B.STORES_HEAD)) for d, dirs, files in os.walk(os.path.join(ad.root, B.STORES_HEAD)) if "RESULT" in files)
    assert len(stores) == 12 == facts["runs"] + facts["n_parked"] + facts_q["runs"] and stores[-1] == f"qwen35_4b-f22it/none/thinking_off/bcbw-qwen35_4b_it-8ba0b48-shared-rep2", stores
    runs_q = _by_view_key(bt_q)
    assert set(runs_q) == {("qwen35_4b_it_shared", "none"), ("qwen35_4b_it_shared", "none@rep2")}
    it = runs_q[("qwen35_4b_it_shared", "none")]
    assert it["extra"]["comparison_kind"] == "it_none_comparison" and it["extra"]["launch_set"]["class"] == "unsteered" and it["transfer_kind"] == "none"
    assert it["extra"]["launched_roster"]["tree_sha"] == "acb74340867cf04c70fee91caf0a95101d1ac0e8" and it["extra"]["served_tree_sha"].startswith("8ba0b48"), "hazard 5: the cross-tree citation, matched by hash"
    assert it["extra"]["standup_id"] == "18ab4860fb70d0ff9b2c9b22022a6f98" and it["extra"]["comparison"]["state"] == "no_contrast"
    # the instruct row's repeat (SHARED_IT_REPEAT_RULE): repeat_floor, no solo twin, the ONE contrast of the group -- its floor against the row's own none
    it_rep = runs_q[("qwen35_4b_it_shared", "none@rep2")]
    e_rep = it_rep["extra"]
    assert e_rep["comparison_kind"] == "repeat_floor" and e_rep["launch_set"]["class"] == "repeat" and e_rep["launch_set"]["tuple_id"] == "it/none@rep2" and e_rep["launch_set"]["selection"] == "it"
    assert e_rep["solo_run_id"] is None and e_rep["solo_store"] is None and e_rep["launch_set"]["anchor_tuple"] is None and e_rep["launch_set"]["repeat"] == {"of": "none", "index": 2, "key": "none@rep2", "store_of": "native/qwen35_4b-f22it/none/thinking_off/bcbw-qwen35_4b_it-8ba0b48-shared"}
    assert e_rep["comparison"]["state"] == B.SOLO_STATE_NO_TWIN and e_rep["comparison"]["contrast_anchor_kind"] is None and e_rep["comparison"]["rule"] == B.SHARED_IT_REPEAT_RULE
    assert e_rep["standup_id"] == it["extra"]["standup_id"] == "18ab4860fb70d0ff9b2c9b22022a6f98" and e_rep["gate"]["comparability_class"] == "same_standup" and e_rep["gate"]["compared_against_standup_ids"] == {"this": e_rep["standup_id"], "prior": e_rep["standup_id"]}
    assert it_rep["producer_run_key_legacy"].endswith("@rep2") and it_rep["run_id"] != it["run_id"] and it_rep["transfer_kind"] == "none" and e_rep["instruct_direct"]["own_none"] is True and e_rep["instruct_direct"]["repeat"] == 2
    assert e_rep["store_suffix_legacy"] == "bcbw-qwen35_4b_it-8ba0b48-shared-rep2" and e_rep["three_numbers"]["producer_record_agrees"] is True and e_rep["anchor"]["repeat_floor_of_own_none"] is True
    assert facts_q["comparisons"]["qwen35_4b_it|none@rep2"]["state"] == B.SOLO_STATE_NO_TWIN and facts_q["contrasts"]["qwen35_4b_it|none@rep2"]["state"] == "contrasted"
    assert [(c["run_id"], c["anchor_run_id"], c["anchor_kind"], c["n_pairs"]) for c in bt_q["run_contrast"]] == [(it_rep["run_id"], it["run_id"], "same_standup_none", n_items)]


# --------------------------------------------------------------------------------------------
# the store-attestation decision of 2026-09-29 (the first REAL mirror, landing 1): three reader rules
# --------------------------------------------------------------------------------------------

def test_rule_1_row_dir_binds_by_its_rows_row_suffix_tail_and_records_the_absolute_spelling_verbatim(tmp_path, seeds):
    """RULE 1 (`SHARED_ROW_DIR_RULE`): `manifest.row_engine.row_dir` may be the volume-absolute `/vol/tinycap/gen/rows/<row>/<suffix>`; the reader binds it by the
    `rows/<row>/<suffix>` TAIL -- the row launch's container: this store's row, the row's primary launch tag, the store's own relaunch segment, no repeat segment -- and
    records the spelling verbatim (never compared); a tail that is not this store's row launch refuses by name, as does a container the launch row declares differently."""
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=True, it_none=True, relaunched=True, row_dir_absolute=True)
    assert facts["runs"] == 5 and facts["n_refusals"] == 0 and facts["n_parked"] == 1, facts["refusals"]
    runs = _by_view_key(bt)
    sfx = fx["solo_suffix"] + "-shared"
    for key, tail in ((("gemma4e4b_s1_shared", NONE), f"rows/{SH.ROW_DIR}/{sfx}"), (("gemma4e4b_s1_shared", "none@rep2"), f"rows/{SH.ROW_DIR}/{sfx}"),
                      (("gemma4e4b_s1_shared", PROMPTED), f"rows/{SH.ROW_DIR}/{sfx}"), (("gemma4e4b_s1_shared", RELU), f"rows/{SH.ROW_DIR}/{sfx}-r2"),
                      (("gemma4e4b_it_shared", NONE), f"rows/{SH.IT_ROW_DIR}/bcbw-{IT_TAG}-{SH.TREE[:7]}-shared")):
        re_ = runs[key]["extra"]["row_engine"]
        assert re_["row_record_dir"] == f"{B.VOLUME_HEAD}{tail}" and re_["row_record_dir_tail"] == tail and re_["row_record_dir_absolute"] is True, (key, re_["row_record_dir"])
        assert re_["row_record_dir_tree_agrees"] is True and re_["row_record_dir_tree_sha7"] == SH.TREE[:7] and re_["row_record_dir_declared"] is None and re_["row_dir_rule"] == B.SHARED_ROW_DIR_RULE
        assert runs[key]["extra"]["gate"]["row_record_dir"] == f"{B.VOLUME_HEAD}{tail}", "the gate record's spelling is recorded verbatim too; the two bind by the tail"
    assert "-rep2" not in runs[("gemma4e4b_s1_shared", "none@rep2")]["extra"]["row_engine"]["row_record_dir"], "a repeat rides its row's container"
    assert runs[("gemma4e4b_s1_shared", RELU)]["extra"]["row_engine"]["row_launch_tag_legacy"] == TAG
    # the relative spelling reads the same way (recorded, not absolute); the pure helpers
    fx0, ad0, bt0, facts0 = _read(tmp_path / "rel", seeds, repeat=False)
    assert all(r["extra"]["row_engine"]["row_record_dir_absolute"] is False and r["extra"]["row_engine"]["row_record_dir"] == r["extra"]["row_engine"]["row_record_dir_tail"] for r in bt0["run"])
    assert B.row_record_tail(f"{B.VOLUME_HEAD}rows/{SH.ROW_DIR}/{sfx}") == B.row_record_tail(f"rows/{SH.ROW_DIR}/{sfx}") == f"rows/{SH.ROW_DIR}/{sfx}"
    assert B.row_record_tail(f"{B.VOLUME_HEAD}native/{SH.ROW_DIR}/{sfx}") is None and B.row_record_tail("rows/x") is None and B.row_record_tail("") is None
    assert B.row_launch_suffix(BCS.SPEC, f"{sfx}-rep2") == sfx and B.row_launch_suffix(BCS.SPEC, f"{sfx}-r2-rep2") == f"{sfx}-r2" and B.row_launch_suffix(BCS.SPEC, f"{sfx}-r2") == f"{sfx}-r2"
    assert B.row_launch_suffix(BCF.SPEC, "bcbw-gemma4e4b_s1-8ba0b48-r2") == "bcbw-gemma4e4b_s1-8ba0b48-r2", "a solo reader's suffix is its own"
    # REFUSED by name: another row, another launch tag, a repeat segment in the container, a spelling outside rows/ -- each on the none store alone; the siblings file
    for bad, fragment in ((f"{B.VOLUME_HEAD}rows/gemma4e4b-f22s2/{sfx}", "does not name this store's row launch"),
                          (f"{B.VOLUME_HEAD}rows/{SH.ROW_DIR}/bcbw-{TAG}_arditi-{SH.TREE[:7]}-shared", "does not name this store's row launch"),
                          (f"rows/{SH.ROW_DIR}/{sfx}-rep2", "does not name this store's row launch"),
                          (f"{B.VOLUME_HEAD}native/{SH.ROW_DIR}/{sfx}", "does not name this store's row launch"),
                          (f"rows/{SH.ROW_DIR}/{fx['solo_suffix']}", "does not name this store's row launch")):
        fxb, adb, btb, factsb = _read(tmp_path / hashlib.md5(bad.encode()).hexdigest()[:8], seeds, repeat=False, row_dir_override=bad)
        why = _refusal(factsb, f"{TAG}|{NONE}")
        assert fragment in why and B.SHARED_ROW_DIR_RULE[:40] in why and repr(bad) in why and factsb["runs"] == 2, (bad, why)
    # the launch row's OWN declaration (the real mirror's rows carry `row_dir`): equal tails read clean and are recorded; a different container refuses
    def declare(doc, wrong_for=None):
        for ev in doc["evals"].values():
            for r in ev["launch"]:
                tail = f"rows/{r['launched_as']['row_dir']}/{B.row_launch_suffix(BCS.SPEC, r['launched_as']['suffix'])}"
                r["row_dir"] = f"{B.VOLUME_HEAD}{tail}" if r["cell"] != wrong_for else f"{B.VOLUME_HEAD}rows/{r['launched_as']['row_dir']}/bcbw-{r['tag']}-0000000-shared"
    fxd, add, btd, factsd = _read(tmp_path / "decl", seeds, repeat=True, relaunched=True, row_dir_absolute=True, launch_override=declare)
    assert factsd["runs"] == 4 and factsd["n_refusals"] == 0, factsd["refusals"]
    assert all(r["extra"]["row_engine"]["row_record_dir_declared"] == r["extra"]["row_engine"]["row_record_dir"] for r in btd["run"])
    fxe, ade, bte, factse = _read(tmp_path / "decl2", seeds, repeat=False, row_dir_absolute=True, launch_override=lambda d: declare(d, wrong_for=NONE))
    why = _refusal(factse, f"{TAG}|{NONE}")
    assert "the store's manifest and its launch row name different row launches" in why and "0000000" in why and factse["runs"] == 2


def test_rule_2_the_served_map_is_the_endpoints_and_a_last_key_endpoint_reads_with_a_downgrade_and_a_note(tmp_path, seeds):
    """RULE 2 (`SHARED_ENDPOINT_TUPLE_RULE`): `manifest.route.object_artifact_sha256_by_cell` may be null while `served_geometry.json:endpoint` carries the row's map,
    and the endpoint's `cell` / `harness_cell` may name the row's LAST served key (landing 1's row runner, `ENDPOINT_SERVED_KEY_BUG`): the served map is the endpoint's,
    the store's tuple the manifest's, both cross-checked against the pinned object and RESULT; where the endpoint names the last key the store reads with
    `field_confidence_json.object_id` downgraded to derived and a source_status note (the status extra); an agreeing endpoint reads clean with no note; a map entry
    that is not the store's object, an endpoint naming any other key, a missing map or two unequal maps refuse by name."""
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=True, it_none=True, manifest_map_null=True, endpoint_last_key=True)
    assert facts["runs"] == 5 and facts["n_refusals"] == 0, facts["refusals"]
    runs = _by_view_key(bt)
    for r in bt["run"]:
        re_ = r["extra"]["row_engine"]
        assert re_["served_map_source"] == B.SERVED_MAP_SOURCE_ENDPOINT == "endpoint" and re_["manifest_route_map_null"] is True and re_["object_artifact_sha256_scalar"] is None
        assert re_["endpoint_tuple_rule"] == B.SHARED_ENDPOINT_TUPLE_RULE and re_["result_tokens"]["standup"] == re_["standup_id"] and re_["result_tokens"]["sha"] == SH.TREE
    # the s1 row's last served key is the prompted key; the it row's the fitted key: the none, the ReLU key, the repeat and the it none carry the note; object_id one step
    # down from its baseline -- the ReLU key's recorded -> derived (the object derived from the manifest, the endpoint's map entry and RESULT), a none's kept at derived
    noted = {k for k, r in runs.items() if r["extra"]["row_engine"]["endpoint_served_key_note"]}
    assert noted == {("gemma4e4b_s1_shared", NONE), ("gemma4e4b_s1_shared", RELU), ("gemma4e4b_s1_shared", "none@rep2"), ("gemma4e4b_it_shared", NONE)}, noted
    for k in noted:
        r, re_ = runs[k], runs[k]["extra"]["row_engine"]
        fc, oc = r["field_confidence_json"]["object_id"], re_["object_id_confidence"]
        assert fc["confidence"] == "derived" == oc["filed"] and "landing 1's row runner" in fc["source"] and "endpoint.cell names" in fc["source"] and "standup= token" in fc["source"], fc
        assert oc == ({"baseline": "recorded", "filed": "derived", "downgraded": True} if k[1] == RELU else {"baseline": "derived", "filed": "derived", "downgraded": False}), (k, oc)
        assert ("downgraded from recorded" in fc["source"]) == (k[1] == RELU) and ("kept at derived" in fc["source"]) == (k[1] != RELU)
        assert re_["endpoint_served_key_agrees"] is False and re_["endpoint_served_key"] == (PROMPTED if k[0] == "gemma4e4b_s1_shared" else Z.IT_KEY) and re_["manifest_served_key"] == k[1].split("@")[0]
        assert r["extra"]["endpoint_served_key_note"] == re_["endpoint_served_key_note"] and B.ENDPOINT_SERVED_KEY_BUG in re_["endpoint_served_key_note"]
        assert re_["endpoint_served_harness_key"] == X.harness_id(re_["endpoint_served_key"]) and re_["manifest_served_harness_key"] == X.harness_id(k[1].split("@")[0])
    assert runs[("gemma4e4b_s1_shared", RELU)]["extra"]["row_engine"]["served_map_entry"] == Z.OBJ == runs[("gemma4e4b_s1_shared", RELU)]["object_id"], "the digest binds through the endpoint's map"
    assert runs[("gemma4e4b_s1_shared", NONE)]["extra"]["row_engine"]["served_map_entry"] is None
    # the prompted store IS the row's last key: its endpoint agrees, no note, object_id as recorded
    clean = runs[("gemma4e4b_s1_shared", PROMPTED)]
    assert clean["extra"]["row_engine"]["endpoint_served_key_agrees"] is True and clean["extra"]["row_engine"]["endpoint_served_key_note"] is None
    assert clean["field_confidence_json"]["object_id"]["confidence"] == "recorded" and clean["extra"]["row_engine"]["object_id_confidence"] == {"baseline": "recorded", "filed": "recorded", "downgraded": False}
    assert "endpoint_served_key_note" not in clean["extra"] and "landing 1" not in clean["field_confidence_json"]["object_id"]["source"]
    # the source_status note (the status extra): one entry per FILED store whose endpoint names the last key, naming the bug and the downgraded field
    assert facts["n_endpoint_served_key_mismatch"] == 4 and {e["row"] for e in facts["endpoint_served_key_mismatch"]} == {f"{TAG}|{NONE}", f"{TAG}|{RELU}", f"{TAG}|none@rep2", f"{IT_TAG}|{NONE}"}
    for e in facts["endpoint_served_key_mismatch"]:
        assert e["downgraded_field"] == "object_id" and e["note"] == B.ENDPOINT_SERVED_KEY_BUG and e["rule"] == B.SHARED_ENDPOINT_TUPLE_RULE and e["run_id"] in {r["run_id"] for r in bt["run"]}
        assert e["endpoint_key"] == e["row_last_key"] == (PROMPTED if e["row"].startswith(TAG) else Z.IT_KEY) and e["manifest_key"] != e["endpoint_key"]
    assert facts["rows"][f"{TAG}|{NONE}"]["endpoint_served_key_agrees"] is False and facts["rows"][f"{TAG}|{PROMPTED}"]["endpoint_served_key_agrees"] is True
    # a clean mirror (every endpoint its own store's; the manifest's map present too): no note, count 0, the map from both sources held equal
    fx2, ad2, bt2, facts2 = _read(tmp_path / "b", seeds, repeat=True)
    assert facts2["n_endpoint_served_key_mismatch"] == 0 and facts2["endpoint_served_key_mismatch"] == [] and facts2["n_refusals"] == 0
    assert all(r["extra"]["row_engine"]["object_id_confidence"]["downgraded"] is False and "landing 1" not in r["field_confidence_json"]["object_id"]["source"]
               and r["extra"]["row_engine"]["served_map_source"] == B.SERVED_MAP_SOURCE_BOTH and r["extra"]["row_engine"]["endpoint_served_key_agrees"] is True
               and "endpoint_served_key_note" not in r["extra"] for r in bt2["run"])
    assert {r["extra"]["row_engine"]["object_id_confidence"]["baseline"] for r in bt2["run"]} == {"recorded", "derived"}, "fitted / prompted keys resolve recorded, the none derived"
    # REFUSED by name -- (a) the endpoint map's entry for the store's harness key is not the pinned object
    def other_digest(okey, ep):
        if okey == RELU:
            ep["object_artifact_sha256_by_cell"][X.harness_id(RELU)] = "0" * 64
    fx3, ad3, bt3, facts3 = _read(tmp_path / "c", seeds, repeat=False, manifest_map_null=True, endpoint_edit=other_digest)
    why = _refusal(facts3, f"{TAG}|{RELU}")
    assert "endpoint.object_artifact_sha256_by_cell" in why and "does not name this store's tuple's object" in why and facts3["runs"] == 2, why
    # (b) an endpoint naming a served key that is neither this store's nor the row's last (the middle key), (c) a key the engine does not serve
    assert "constv2ctl2-full__lora_r4_ce_minus_ctl" not in SH.ROW_KEYS and RELU == SH.ROW_KEYS[1]
    for sub_dir, okey_named, label in (("d1", RELU, "the middle key"), ("d2", "constv2ctl2-full__lora_r4_ce_minus_ctl", "a key the engine does not serve")):
        def elsewhere(okey, ep, named=okey_named):
            if okey == NONE:
                ep["cell"], ep["harness_cell"] = named, X.harness_id(named)
        fx4, ad4, bt4, facts4 = _read(tmp_path / sub_dir, seeds, repeat=False, endpoint_edit=elsewhere)
        why = _refusal(facts4, f"{TAG}|{NONE}")
        assert "neither this store's tuple" in why and "the row's last served key" in why and B.ENDPOINT_SERVED_KEY_BUG[:30] in why and facts4["runs"] == 2, (label, why)
    # (d) a last-key endpoint whose harness_cell does not follow: not the landing-1 shape either
    def half_bug(okey, ep):
        if okey == NONE:
            ep["cell"] = PROMPTED     # harness_cell stays the none's
    fx5, ad5, bt5, facts5 = _read(tmp_path / "e", seeds, repeat=False, endpoint_edit=half_bug)
    assert "neither this store's tuple" in _refusal(facts5, f"{TAG}|{NONE}") and facts5["runs"] == 2
    # (e) no map anywhere, (f) two maps that disagree
    def no_map(okey, ep):
        if okey == NONE:
            ep.pop("object_artifact_sha256_by_cell")
    fx6, ad6, bt6, facts6 = _read(tmp_path / "f", seeds, repeat=False, manifest_map_null=True, endpoint_edit=no_map)
    assert "carries no object_artifact_sha256_by_cell map" in _refusal(facts6, f"{TAG}|{NONE}") and facts6["runs"] == 2
    def unequal(okey, ep):
        if okey == NONE:
            ep["object_artifact_sha256_by_cell"] = dict(ep["object_artifact_sha256_by_cell"], si_other="1" * 64)
    fx7, ad7, bt7, facts7 = _read(tmp_path / "g", seeds, repeat=False, endpoint_edit=unequal)
    assert "both present, they must be one map" in _refusal(facts7, f"{TAG}|{NONE}") and facts7["runs"] == 2


def test_rule_3_the_instruct_rows_repeat_none_is_a_repeat_floor_never_it_none_comparison(tmp_path, seeds):
    """RULE 3 (`SHARED_IT_REPEAT_RULE`): an instruct row's `none@rep2` store (`<model key>_it|none@rep2`, suffix `...-shared-rep2`) is a repeat_floor row -- the repeat
    segment is read BEFORE the instruct-none classification -- admitted under class repeat with no solo run and no solo store (the `_it` none has no solo twin:
    `no_solo_twin`, no borrowed contrast), its in-run floor against the row's own none (the it_none_comparison run, same_standup_none: the one contrast that run
    anchors); never refused and never filed as it_none_comparison. Before the rule the reader classified the store as the instruct row's own none and refused the
    launch row's `repeat_floor` as the wrong kind."""
    fx, ad, bt, facts = _read(tmp_path, seeds, repeat=False, it_repeat=True)
    assert facts["runs"] == 5 and facts["n_refusals"] == 0 and facts["not_run"] == [], facts["refusals"]
    runs = _by_view_key(bt)
    assert set(runs) == {("gemma4e4b_s1_shared", NONE), ("gemma4e4b_s1_shared", RELU), ("gemma4e4b_s1_shared", PROMPTED), ("gemma4e4b_it_shared", NONE), ("gemma4e4b_it_shared", "none@rep2")}
    it_none, it_rep = runs[("gemma4e4b_it_shared", NONE)], runs[("gemma4e4b_it_shared", "none@rep2")]
    e = it_rep["extra"]
    inst = fx["instruments"]["bigcodebench"]
    assert e["comparison_kind"] == "repeat_floor" and e["launch_set"]["class"] == "repeat" and e["launch_set"]["tuple_id"] == "it/none@rep2" and e["launch_set"]["repeat"]["key"] == "none@rep2"
    assert e["solo_run_id"] is None and e["solo_store"] is None and e["comparison"]["state"] == B.SOLO_STATE_NO_TWIN == "no_solo_twin" and e["comparison"]["contrast_anchor_kind"] is None
    assert e["comparison"]["rule"] == B.SHARED_IT_REPEAT_RULE and "no solo twin" in e["comparison"]["note"]
    assert it_rep["producer_run_key_legacy"] == f"gemma4e4b_it_shared/{fx['it_key']}@rep2" and it_rep["run_id"] == BCS.run_id_for(IT_TAG, fx["it_key"], NONE, inst, repeat=2) != it_none["run_id"]
    assert it_rep["transfer_kind"] == "none" and it_rep["object_id"] == "none" and it_rep["view_legacy"] == "gemma4e4b_it_shared" and it_rep["seed_base"] is None
    assert e["instruct_direct"]["own_none"] is True and e["instruct_direct"]["repeat"] == 2 and e["instruct_direct"]["rule"] == B.SHARED_IT_REPEAT_RULE and e["instruct_direct"]["transfer_kind_recorded"] == "none"
    assert e["shared_standup"]["repeat"] == 2 and e["shared_standup"]["row_key"] == "none@rep2" and e["standup_id"] == SH.STANDUP_IT == it_none["extra"]["standup_id"] and "anchor_alias" not in e
    assert e["gate"]["comparison"]["comparison_kind"] == "repeat_floor" and e["gate"]["comparability_class"] == "same_standup" and e["three_numbers"]["producer_record_agrees"] is True
    assert "repeat floor" in it_rep["notes"] and "repeat 2" in it_rep["notes"]
    # the in-run floor: against the row's OWN none, same_standup_none; no borrowed contrast; the it none anchors nothing else and forms no contrast of its own
    floor = [c for c in bt["run_contrast"] if c["run_id"] == it_rep["run_id"]]
    assert len(floor) == 1 and floor[0]["anchor_run_id"] == it_none["run_id"] and floor[0]["anchor_kind"] == "same_standup_none" and floor[0]["delta"] == 0.0 and floor[0]["computed_by"] == "registry"
    assert floor[0]["extra"]["repeat_floor_of_own_none"] is True and floor[0]["extra"]["comparison_kind"] == "repeat_floor" and floor[0]["extra"]["anchor_rule"] == B.SHARED_IT_REPEAT_RULE
    assert floor[0]["extra"]["standup_ids"] == {"row": SH.STANDUP_IT, "anchor": SH.STANDUP_IT} and floor[0]["extra"]["cross_standup"] is False and "no solo twin" in floor[0]["extra"]["floor_note"]
    assert [c for c in bt["run_contrast"] if c["anchor_run_id"] == it_none["run_id"]] == floor and not [c for c in bt["run_contrast"] if c["run_id"] == it_none["run_id"]]
    assert not [c for c in _contrasts(bt, "solo_stand_in") if c["run_id"] in (it_rep["run_id"], it_none["run_id"])] and len(_contrasts(bt, "solo_stand_in")) == 3
    assert e["anchor"]["state"] == "contrasted" and e["anchor"]["anchor_run_id"] == it_none["run_id"] and e["anchor"]["repeat_floor_of_own_none"] is True and e["anchor"]["rule"] == B.SHARED_IT_REPEAT_RULE
    assert e["anchor"]["anchor_view_legacy"] == IT_TAG and e["anchor"]["cross_standup"] is False
    assert facts["comparisons"][f"{IT_TAG}|none@rep2"]["state"] == "no_solo_twin" and facts["contrasts"][f"{IT_TAG}|none@rep2"]["state"] == "contrasted" and facts["contrasts"][f"{IT_TAG}|none@rep2"]["repeat_floor_of_own_none"] is True
    assert facts["rows"][f"{IT_TAG}|none@rep2"]["it_repeat"] is True and facts["rows"][f"{IT_TAG}|none@rep2"]["it_none"] is False and facts["rows"][f"{IT_TAG}|none@rep2"]["repeat"] == 2
    assert "instruct_direct" not in facts["rows"][f"{IT_TAG}|none@rep2"] and facts["standups"][SH.STANDUP_IT]["rows"] == [f"{IT_TAG}|{NONE}", f"{IT_TAG}|none@rep2"]
    # the it none itself is what it was: it_none_comparison, no contrast of its own
    assert it_none["extra"]["comparison_kind"] == "it_none_comparison" and it_none["extra"]["comparison"]["state"] == "no_contrast" and it_none["extra"]["launch_set"]["class"] == "unsteered"
    assert facts["rows"][f"{IT_TAG}|{NONE}"]["it_none"] is True and facts["rows"][f"{IT_TAG}|{NONE}"]["it_repeat"] is False
    # the source states name the repeat by tuple id
    states = ad.source_states(X.ctx(), ad.discover())
    ls = next(s for s in states if s.input_uri.endswith(f"launch_sets_shared.json#{GROUP}"))
    assert "it/none@rep2" in ls.current_members and "it/none" in ls.current_members
    # REFUSED by name: the launch row filing the instruct repeat as it_none_comparison (the kind follows the repeat segment), or naming a solo run for it (no solo twin)
    def relabel(doc):
        for ev in doc["evals"].values():
            for r in ev["launch"]:
                if r["tag"] == IT_TAG and r["tuple_id"].endswith("@rep2"):
                    r["comparison_kind"] = "it_none_comparison"
    fx2, ad2, bt2, facts2 = _read(tmp_path / "b", seeds, repeat=False, it_repeat=True, launch_override=relabel)
    why = _refusal(facts2, f"{IT_TAG}|none@rep2")
    assert "comparison_kind 'it_none_comparison'; this store is 'repeat_floor'" in why and "its repeat floor" in why and facts2["runs"] == 4, why
    def name_a_solo(doc):
        for ev in doc["evals"].values():
            for r in ev["launch"]:
                if r["tag"] == IT_TAG and r["tuple_id"].endswith("@rep2"):
                    r["solo_run_id"] = "sdfbase_full22:bigcodebench:461f34836b51:0000000000000000"
    fx3, ad3, bt3, facts3 = _read(tmp_path / "c", seeds, repeat=False, it_repeat=True, launch_override=name_a_solo)
    why = _refusal(facts3, f"{IT_TAG}|none@rep2")
    assert "no solo twin" in why and B.SHARED_IT_REPEAT_RULE[:40] in why and facts3["runs"] == 4, why


# --------------------------------------------------------------------------------------------
# landing 2: the `_l2` sources -- the same tuples, another harness, distinct run ids; the verified store's standup rule
# --------------------------------------------------------------------------------------------

def test_the_l2_specs_bind_the_l2_roots_and_the_producer_key_carries_the_harness_variant(seeds):
    """The `_l2` readers are landing 1's readers parametrized once more: their own root and source, the same spelling (`<tag>_shared`), and a producer key whose
    SOURCE segment carries the harness variant (`<tag>_shared_l2/<store key>[@rep<N>]`, SHARED_L2_RUN_ID_RULE) -- so every run id differs from landing 1's and
    from the solo reader's over one store key while the campaign / eval / instrument prefix is the same. Landing 1's spec carries none of it."""
    key, inst = "none__controls__t0__a0__2ff9cdff__1dd7b444", "bigcodebench_full_protocol_greedy_4096_synthetic"
    for l2, l1, solo in ((BCS2, BCS, BCF), (BFS2, BFS, BFF)):
        s = l2.SPEC
        assert s.source == l2.SOURCE == l2.NAME == f"{solo.SOURCE}_shared_l2" == f"{l1.SOURCE}_l2" and s.root_default.endswith(os.path.join("run-state", "full22", l2.SOURCE))
        assert s.root_default not in (l1.SPEC.root_default, solo.SPEC.root_default) and l2.ROOT_DEFAULT == s.root_default
        assert s.shared is not None and s.shared.harness_variant == "l2" == B.SHARED_L2_HARNESS_VARIANT and s.shared.base_source == l1.SOURCE == l2.BASE_SOURCE
        assert s.shared.solo_source == solo.SOURCE and l2.SOLO_ROOT_DEFAULT == l1.SOLO_ROOT_DEFAULT == solo.ROOT_DEFAULT
        # everything else is landing 1's: the records' names, the eval, the item set and instrument, the model set, the grammar
        for f in ("eval_id", "task", "record_task", "outcomes_file", "instrument_seed", "cap", "protocol_temperature", "item_set_key", "item_set_id", "item_set_label", "item_set_pin_file",
                  "roster_file", "models", "instruct_direct", "clamp_override", "tool_calls_required", "n_items", "three_numbers_file", "launch_set_files", "schedule_file"):
            assert getattr(s, f) == getattr(l1.SPEC, f), f
        assert s.suffix_re.pattern == l1.SPEC.suffix_re.pattern and s.grammar_text == l1.SPEC.grammar_text
        # the spelling is landing 1's; the producer key's spelling segment is not
        assert l2.spelling_of("gemma4e4b_s1") == l1.spelling_of("gemma4e4b_s1") == "gemma4e4b_s1_shared" and l2.producer_spelling_of("gemma4e4b_s1") == "gemma4e4b_s1_shared_l2"
        assert B.shared_producer_key_for(s, "gemma4e4b_s1", key) == f"gemma4e4b_s1_shared_l2/{key}" and B.shared_producer_key_for(s, "gemma4e4b_s1", key, 2) == f"gemma4e4b_s1_shared_l2/{key}@rep2"
        assert B.shared_producer_key_for(l1.SPEC, "gemma4e4b_s1", key) == f"gemma4e4b_s1_shared/{key}" and B.shared_producer_key_for(solo.SPEC, "gemma4e4b_s1", key) == f"gemma4e4b_s1/{key}"
        ids = {"l2": l2.run_id_for("gemma4e4b_s1", key, "none", inst), "l1": l1.run_id_for("gemma4e4b_s1", key, "none", inst), "solo": solo.run_id_for("gemma4e4b_s1", key, "none", inst),
               "l2_rep": l2.run_id_for("gemma4e4b_s1", key, "none", inst, repeat=2), "l1_rep": l1.run_id_for("gemma4e4b_s1", key, "none", inst, repeat=2)}
        assert len(set(ids.values())) == 5, ids
        assert l2.landing1_run_id_for("gemma4e4b_s1", key, "none", inst) == ids["l1"] and l2.solo_run_id_for("gemma4e4b_s1", key, "none", inst) == ids["solo"]
        parsed = {k: I.parse_run_id(v) for k, v in ids.items()}
        assert len({(p["campaign"], p["eval_id"], p["instrument_prefix"]) for p in parsed.values()}) == 1, "the discriminator is the producer key, never the campaign / eval / instrument"
        assert len({p["producer_sha16"] for p in parsed.values()}) == 5
        # landing 1's spec is untouched: no harness variant, the producer spelling IS the registry spelling
        assert l1.SHARED.harness_variant is None and l1.SHARED.base_source is None and l1.SHARED.producer_spelling_of("x_s1") == l1.spelling_of("x_s1") == "x_s1_shared"
        assert l2.recount_computation_id() not in (l1.recount_computation_id(), solo.recount_computation_id()) and l2.RECOUNT_ALGORITHM.endswith(f"{l2.NAME}.recount")


@pytest.mark.parametrize("eval_id", ["bigcodebench", "bfcl"])
def test_l2_stores_read_under_the_l2_source_with_run_ids_distinct_from_landing_1_and_the_base_own_source_recorded(tmp_path, seeds, eval_id):
    """The same stores under the `_l2` root file on the `_l2` source with landing 1's spellings and contrasts, run ids disjoint from the landing-1 reader's over the
    SAME store keys, the producer key `<tag>_shared_l2/...` recorded, and the bound launch set's evals.<eval>.own_source spelling the BASE source recorded as provenance
    (`extra.launch_set.own_source_spelled`; the status extra), never refused -- the reader binds by root (SHARED_L2_RUN_ID_RULE)."""
    fx2, ad2, bt2, facts2 = _read(tmp_path / "l2", seeds, eval_id=eval_id, landing=2, verified=True)
    fx1, ad1, bt1, facts1 = _read(tmp_path / "l1", seeds, eval_id=eval_id)
    assert isinstance(ad2, SH.ADAPTERS_L2[eval_id]) and isinstance(ad1, SH.ADAPTERS[eval_id]) and fx2["keys"] == fx1["keys"]        # the same stores (the same keys)
    assert facts2["n_refusals"] == 0 and facts2["runs"] == 4 == facts1["runs"], facts2["refusals"]
    r2, r1 = SH.runs_by_key(bt2), SH.runs_by_key(bt1)
    assert set(r2) == set(r1) == {NONE, RELU, PROMPTED, f"{NONE}@rep2"}
    assert not ({r["run_id"] for r in r2.values()} & {r["run_id"] for r in r1.values()}), "no landing-2 id equals a landing-1 id"
    l2, l1 = SH.SHARED_L2[eval_id], SH.SHARED[eval_id]
    for rk, r in r2.items():
        okey, rep = B.split_row_okey(rk)
        assert r["lane_source_legacy"] == l2.SOURCE and r["view_legacy"] == r1[rk]["view_legacy"] == f"{TAG}_shared" and r["model_id"] == r1[rk]["model_id"]
        assert r["producer_run_key_legacy"] == f"{TAG}_shared_l2/{fx2['keys'][okey]}{'@rep2' if rep else ''}" and r1[rk]["producer_run_key_legacy"] == f"{TAG}_shared/{fx1['keys'][okey]}{'@rep2' if rep else ''}"
        assert r["run_id"] == l2.run_id_for(TAG, fx2["keys"][okey], okey, fx2["instruments"][eval_id], repeat=rep) and r["run_id"] != l1.run_id_for(TAG, fx2["keys"][okey], okey, fx2["instruments"][eval_id], repeat=rep)
        ex = r["extra"]
        assert ex["solo_run_id"] == r1[rk]["extra"]["solo_run_id"] == fx2["solo_run_ids"][okey][eval_id] and ex["comparison_kind"] == r1[rk]["extra"]["comparison_kind"]
        assert ex["shared_standup"]["harness_variant"] == "l2" and ex["shared_standup"]["base_source"] == l1.SOURCE and ex["shared_standup"]["producer_spelling"] == f"{TAG}_shared_l2"
        assert ex["shared_standup"]["spelling"] == f"{TAG}_shared" and ex["shared_standup"]["source_binding"] == "root" and ex["shared_standup"]["run_id_rule"] == B.SHARED_L2_RUN_ID_RULE
        assert ex["launch_set"]["own_source_spelled"] == l1.SOURCE and ex["launch_set"]["own_source_is_base"] is True and ex["launch_set"]["spelling"] == f"{TAG}_shared"
        assert ex["source_legacy"] == l2.SOURCE
        assert "harness_variant" not in r1[rk]["extra"]["shared_standup"] and "own_source_spelled" not in r1[rk]["extra"]["launch_set"]      # landing 1's rows carry none of it
    assert facts2["harness_variant"] == "l2" and facts2["base_source"] == l1.SOURCE and facts2["launch_set_own_source_spelled"] == l1.SOURCE and facts2["launch_set_own_source_is_base"] is True
    assert facts2["source_binding"] == "root" and "harness_variant" not in facts1
    # the contrasts are landing 1's: the borrowed comparison against the solo run, the in-run contrast against the row's shared none, the repeat's floor
    assert {c["anchor_kind"] for c in _contrasts(bt2)} == {c["anchor_kind"] for c in _contrasts(bt1)} == {B.ANCHOR_KIND_SOLO_STAND_IN, "same_standup_none"}
    assert len(_contrasts(bt2, B.ANCHOR_KIND_SOLO_STAND_IN)) == len(_contrasts(bt1, B.ANCHOR_KIND_SOLO_STAND_IN)) == 4 and len(_contrasts(bt2, "same_standup_none")) == 3
    assert {c["anchor_run_id"] for c in _contrasts(bt2, B.ANCHOR_KIND_SOLO_STAND_IN)} == set(fx2["solo_run_ids"][k][eval_id] for k in (NONE, RELU, PROMPTED))
    # the launch set spelling the `_l2` name itself reads too: the own_source is recorded either way, never bound
    fx3, ad3, bt3, facts3 = _read(tmp_path / "l2own", seeds, eval_id=eval_id, landing=2, verified=True, own_source_override=l2.SOURCE)
    assert facts3["n_refusals"] == 0 and facts3["runs"] == 4 and facts3["launch_set_own_source_spelled"] == l2.SOURCE and facts3["launch_set_own_source_is_base"] is False
    assert {r["run_id"] for r in bt3["run"]} == {r["run_id"] for r in bt2["run"]}


def test_a_verified_store_reads_the_standup_id_from_the_manifest_when_the_re_stamp_last_line_carries_no_token(tmp_path, seeds):
    """INTEGRATION through the real adapter (2026-09-30): a fixture store shaped like a landing-2 store -- RESULT's last line the CPU
    verifier's re-stamp `result=COMPLETE status=OK sha=<tree> verified_by=<fcid>` WITHOUT standup=, the row runner's earlier GENERATED line WITH it,
    manifest.row_engine.verification_pass.state verified -- files with the MANIFEST's standup id (SHARED_VERIFIED_STANDUP_RULE), the verifier's call and where the
    token was read recorded; the same shape reads through landing 1's reader too (the rule is the store's, not the source's); a store whose RESULT carries no
    standup= token at all reads the manifest's alone."""
    fx, ad, bt, facts = _read(tmp_path, seeds, landing=2, verified=True)
    assert facts["n_refusals"] == 0 and facts["runs"] == 4, facts["refusals"]
    none_dir = os.path.join(fx["root"], B.STORES_HEAD, SH.ROW_DIR, NONE, B.THINKING_DIR, f"{fx['solo_suffix']}-shared")
    with open(os.path.join(none_dir, "RESULT"), encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    assert len(lines) == 2 and "standup=" not in lines[-1] and f"verified_by={SH.VERIFIER_FC}" in lines[-1] and "result=COMPLETE status=OK" in lines[-1]
    assert f"standup={SH.STANDUP_A}" in lines[0] and "result=GENERATED" in lines[0], lines
    for rk, r in SH.runs_by_key(bt).items():
        re_blk = r["extra"]["row_engine"]
        assert r["extra"]["standup_id"] == re_blk["standup_id"] == SH.STANDUP_A == r["extra"]["shared_standup"]["standup_id"]
        assert re_blk["verification_state"] == "verified" and re_blk["verification_pass"]["state"] == "verified" and re_blk["verification_pass"]["function_call_id"] == SH.VERIFIER_FC
        assert re_blk["result_standup"] == "" and re_blk["result_tokens"]["standup"] == "" and re_blk["result_standup_earlier"] == [SH.STANDUP_A]
        assert re_blk["result_standup_source"].startswith("earlier line") and re_blk["result_verified_by"] == SH.VERIFIER_FC == re_blk["verification_function_call_id"]
        assert re_blk["verified_standup_rule"] == B.SHARED_VERIFIED_STANDUP_RULE and re_blk["rule"] == B.SHARED_STANDUP_RULE
        assert r["extra"]["result_line"] == lines[-1].replace(f"-{B.SHARED_STORE_MARKER}-rep2", f"-{B.SHARED_STORE_MARKER}") if rk == NONE else "verified_by=" in r["extra"]["result_line"]
        assert r["extra"]["gate"]["verification_pass"]["state"] == "verified" and r["extra"]["attempts_before_accepted"] == [lines[0]] if rk == NONE else True
        assert r["run_id"] != fx["solo_run_ids"][B.split_row_okey(rk)[0]]["bigcodebench"]
    # the standup fact and the three-number cross-check hold: one standup, four rows, the producer's result_line the re-stamp
    assert set(facts["standups"]) == {SH.STANDUP_A} and len(facts["standups"][SH.STANDUP_A]["rows"]) == 4
    # landing 1's reader over the same shape: the rule is on the store (verification_pass verified), so it reads there too, byte for byte otherwise
    fx1, ad1, bt1, facts1 = _read(tmp_path / "l1v", seeds, landing=1, verified=True)
    assert isinstance(ad1, BCS.BigcodebenchFullSharedAdapter) and facts1["n_refusals"] == 0 and facts1["runs"] == 4
    assert all(r["extra"]["row_engine"]["result_standup_source"].startswith("earlier line") for r in bt1["run"])
    # no RESULT line carries the token at all: the manifest's alone, said so
    fx2, ad2, bt2, facts2 = _read(tmp_path / "nostandup", seeds, landing=2, verified=True, verified_result_no_standup=True)
    assert facts2["n_refusals"] == 0 and facts2["runs"] == 4
    none_run = SH.runs_by_key(bt2)[NONE]["extra"]["row_engine"]
    assert none_run["result_standup_source"].startswith("manifest only") and none_run["result_standup_earlier"] == [] and none_run["standup_id"] == SH.STANDUP_A
    assert SH.runs_by_key(bt2)[RELU]["extra"]["row_engine"]["result_standup_source"].startswith("earlier line")


def test_a_verified_store_whose_result_disagrees_with_the_manifest_refuses_by_name_and_an_unverified_store_keeps_the_last_line_rule(tmp_path, seeds):
    """A PRESENT standup= token that differs from the manifest's refuses by name, verified or not (the last line's under the landing-1 rule; the row runner's earlier
    line under the verified rule); a re-stamp whose verified_by is not verification_pass.function_call_id refuses; a store whose verification_pass is not `verified`
    and whose last line lacks the token refuses under the landing-1 rule (the absent token is admitted on a verified store alone)."""
    # the row runner's line spells another standup on a verified store
    fx, ad, bt, facts = _read(tmp_path / "a", seeds, landing=2, verified=True, standup_override=hashlib.md5(b"another standup").hexdigest())
    why = _refusal(facts, f"{TAG}|{NONE}")
    assert "RESULT's earlier line(s) carry standup=" in why and B.SHARED_VERIFIED_STANDUP_RULE[:40] in why and facts["runs"] == 3
    # the re-stamp names another verifier call
    fx2, ad2, bt2, facts2 = _read(tmp_path / "b", seeds, landing=2, verified=True, verified_by_override="fc-01SYNTHETICOTHERVERIFIER0000")
    why2 = _refusal(facts2, f"{TAG}|{NONE}")
    assert "verified_by='fc-01SYNTHETICOTHERVERIFIER0000'" in why2 and "verification_pass.function_call_id" in why2 and facts2["runs"] == 3
    # a PRESENT differing token on the LAST line of a verified store: the landing-1 refusal, verified or not
    fx3 = SH.mirror(tmp_path / "c", seeds, eval_id="bigcodebench", landing=2, verified=True)
    none_dir = os.path.join(fx3["root"], B.STORES_HEAD, SH.ROW_DIR, NONE, B.THINKING_DIR, f"{fx3['solo_suffix']}-shared")
    with open(os.path.join(none_dir, "RESULT"), encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    lines[-1] += f" standup={hashlib.md5(b'x').hexdigest()}"
    X.write(os.path.join(none_dir, "RESULT"), "\n".join(lines) + "\n")
    out3, _ = SH.read_all(SH.adapter("bigcodebench", fx3, seeds))
    bt3, facts3 = out3[GROUP]
    why3 = _refusal(facts3, f"{TAG}|{NONE}")
    assert "RESULT's last line carries standup=" in why3 and "not the manifest's row_engine.standup_id" in why3 and B.SHARED_STANDUP_RULE[:30] in why3 and facts3["runs"] == 3
    # a PRESENT AGREEING token on the last line of a verified store (the later re-stamps) reads, the token read off the last line
    fx4 = SH.mirror(tmp_path / "d", seeds, eval_id="bigcodebench", landing=2, verified=True)
    none_dir4 = os.path.join(fx4["root"], B.STORES_HEAD, SH.ROW_DIR, NONE, B.THINKING_DIR, f"{fx4['solo_suffix']}-shared")
    with open(os.path.join(none_dir4, "RESULT"), encoding="utf-8") as fh:
        lines4 = [ln.rstrip("\n") for ln in fh if ln.strip()]
    lines4[-1] = lines4[-1].replace(" verified_by=", f" standup={SH.STANDUP_A} verified_by=", 1)
    X.write(os.path.join(none_dir4, "RESULT"), "\n".join(lines4) + "\n")
    tn_path = os.path.join(fx4["root"], B.RECORDS_DIR, BCS2.THREE_NUMBERS_FILE)
    tn = json.load(open(tn_path, encoding="utf-8"))
    tn["rows"][f"{TAG}/{NONE}"]["result_line"] = lines4[-1]
    X.write(tn_path, tn)
    out4, _ = SH.read_all(SH.adapter("bigcodebench", fx4, seeds))
    bt4, facts4 = out4[GROUP]
    assert facts4["n_refusals"] == 0 and facts4["runs"] == 4, facts4["refusals"]
    assert SH.runs_by_key(bt4)[NONE]["extra"]["row_engine"]["result_standup_source"] == "last line" and SH.runs_by_key(bt4)[NONE]["extra"]["row_engine"]["result_standup"] == SH.STANDUP_A
    # verification_pass present but NOT verified, the last line without the token: the landing-1 rule refuses (the absent token is admitted on a verified store alone)
    fx5, ad5, bt5, facts5 = _read(tmp_path / "e", seeds, landing=2, verified=True, verified_state_override="pending")
    assert all("RESULT's last line carries standup='<none>'" in x and B.SHARED_STANDUP_RULE[:30] in x for x in facts5["refusals"]) and facts5["runs"] == 0 and len(facts5["refusals"]) == 4
    # a landing-1 store (verification_pass null) whose last line lacks the token refuses the same way -- the rule landing 1 always had
    fx6 = SH.mirror(tmp_path / "f", seeds, eval_id="bigcodebench")
    none_dir6 = os.path.join(fx6["root"], B.STORES_HEAD, SH.ROW_DIR, NONE, B.THINKING_DIR, f"{fx6['solo_suffix']}-shared")
    with open(os.path.join(none_dir6, "RESULT"), encoding="utf-8") as fh:
        lines6 = [ln.rstrip("\n") for ln in fh if ln.strip()]
    lines6[-1] = lines6[-1].split(" standup=")[0]
    X.write(os.path.join(none_dir6, "RESULT"), "\n".join(lines6) + "\n")
    out6, _ = SH.read_all(SH.adapter("bigcodebench", fx6, seeds))
    bt6, facts6 = out6[GROUP]
    assert "RESULT's last line carries standup='<none>'" in _refusal(facts6, f"{TAG}|{NONE}") and facts6["runs"] == 3


def test_the_l2_readers_are_registered_declared_and_versioned():
    from sisteer.registry import adapters as ADP
    reg = ADP.load_all()
    for mod, cls, l1 in ((BCS2, BCS2.BigcodebenchFullSharedL2Adapter, BCS), (BFS2, BFS2.BfclFullSharedL2Adapter, BFS)):
        assert mod.NAME in reg and isinstance(reg[mod.NAME](), cls) and cls.covers_sources == (mod.SOURCE,) and AB.owner_of(cls()) == "Overflow" == mod.OWNER
        assert AB.emission_version_of(cls) == "3" and cls.SPEC.shared is not None and cls.SPEC.shared.harness_variant == "l2" and mod.BASE_SOURCE == l1.SOURCE   # 3: solo_stand_in (migration 0027)
        assert cls(root="/nonexistent/x").status_sources("gemma4e4b_s1") == (mod.SOURCE,) and cls(root="/nonexistent/x").solo_root == l1.SOLO_ROOT_DEFAULT
        assert mod.SOURCE not in l1.BigcodebenchFullSharedAdapter.covers_sources if l1 is BCS else mod.SOURCE not in l1.BfclFullSharedAdapter.covers_sources
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "registry", "sources_r2_v1.yaml"), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = {r["source"]: r["owner"] for r in data["sources"]}
    adds = {a["source"]: a for a in data["meta"]["additions"]}
    for s in (BCS2.SOURCE, BFS2.SOURCE):
        assert names[s] == "the authors" and adds[s]["provisional"] is True and adds[s]["paper_build_commit"] is None and "COMPARISON" in adds[s]["why"]
        assert "collide" in adds[s]["why"] or "landing 1" in adds[s]["why"].lower()


def test_a_three_numbers_record_naming_another_source_under_the_root_refuses_every_row_by_name(tmp_path, seeds):
    """The landing-2 readers' source rule (THREE_NUMBERS_SOURCE_RULE): the root binds the source and mints the producer keys, landing 2's launch set
    legitimately spells landing 1 under own_source, and an unverified landing-1 store passes the standup rule -- so a landing-2 root accidentally populated with landing-1
    stores + records would file duplicate landing-1 measurements under new `_l2` ids. The producer's three-number record is the one file that NAMES the source: its
    top-level `source` must equal the reader's before any row files. A cross-root record swap (the SAME file name in both landings) refuses every row by name -- the
    file, the field `source`, both names -- in both directions and on the solo reader too; the real-shaped fixtures still read."""
    l1, l2 = SH.SHARED["bigcodebench"], SH.SHARED_L2["bigcodebench"]
    fx2 = SH.mirror(tmp_path / "l2", seeds, eval_id="bigcodebench", landing=2, verified=True)
    fx1 = SH.mirror(tmp_path / "l1", seeds, eval_id="bigcodebench")
    p2 = os.path.join(fx2["root"], B.RECORDS_DIR, l2.THREE_NUMBERS_FILE)
    p1 = os.path.join(fx1["root"], B.RECORDS_DIR, l1.THREE_NUMBERS_FILE)
    assert l1.THREE_NUMBERS_FILE == l2.THREE_NUMBERS_FILE and json.load(open(p2))["source"] == l2.SOURCE and json.load(open(p1))["source"] == l1.SOURCE
    keep2, keep1 = open(p2, "rb").read(), open(p1, "rb").read()
    # landing 1's record under the landing-2 root: every row refuses, nothing files under a `_l2` id
    with open(p2, "wb") as fh:
        fh.write(keep1)
    out, _ = SH.read_all(SH.adapter("bigcodebench", fx2, seeds))
    bt, facts = out[GROUP]
    assert facts["runs"] == 0 and not bt.get("run") and len(facts["refusals"]) == 4, facts["refusals"]
    why = _refusal(facts, f"{TAG}|{NONE}")
    assert f"{l2.THREE_NUMBERS_FILE}: the three-numbers record spells source '{l1.SOURCE}'; this reader is '{l2.SOURCE}' -- wrong root or wrong mirror" in why
    assert B.THREE_NUMBERS_SOURCE_RULE[:40] in why and facts["three_numbers_source"] == l1.SOURCE
    assert all(f"spells source '{l1.SOURCE}'; this reader is '{l2.SOURCE}'" in x for x in facts["refusals"])
    # the reverse: landing 2's record under the landing-1 root
    with open(p1, "wb") as fh:
        fh.write(keep2)
    out1, _ = SH.read_all(SH.adapter("bigcodebench", fx1, seeds))
    bt1, facts1 = out1[GROUP]
    assert facts1["runs"] == 0 and len(facts1["refusals"]) == 4 and f"spells source '{l2.SOURCE}'; this reader is '{l1.SOURCE}' -- wrong root or wrong mirror" in _refusal(facts1, f"{TAG}|{NONE}")
    assert "three_numbers_source" not in facts1        # the fact rides the landing-2 readers' status alone: landing 1's status stays byte for byte
    # restored, both read as before (the real-shaped fixtures are unchanged by the rule)
    with open(p2, "wb") as fh:
        fh.write(keep2)
    with open(p1, "wb") as fh:
        fh.write(keep1)
    out2, _ = SH.read_all(SH.adapter("bigcodebench", fx2, seeds))
    out1b, _ = SH.read_all(SH.adapter("bigcodebench", fx1, seeds))
    assert out2[GROUP][1]["runs"] == 4 == out1b[GROUP][1]["runs"] and out2[GROUP][1]["n_refusals"] == 0 == out1b[GROUP][1]["n_refusals"]
    assert out2[GROUP][1]["three_numbers_source"] == l2.SOURCE and "three_numbers_source" not in out1b[GROUP][1]
    # the solo reader holds the same rule (the record under its root spelling another source), with no fact of its own (its status stays byte for byte)
    solo_p = os.path.join(fx1["solo_root"], B.RECORDS_DIR, BCF.THREE_NUMBERS_FILE)
    solo_doc = json.load(open(solo_p, encoding="utf-8"))
    solo_keep = open(solo_p, "rb").read()
    solo_doc["source"] = l1.SOURCE
    X.write(solo_p, solo_doc)
    bts, factss = Z.read(SH.solo_adapter("bigcodebench", fx1, seeds))
    assert factss["runs"] == 0 and factss["n_refusals"] >= 1 and all(f"spells source '{l1.SOURCE}'; this reader is '{BCF.SOURCE}'" in x for x in factss["refusals"]), factss["refusals"][:2]
    assert "three_numbers_source" not in factss
    with open(solo_p, "wb") as fh:
        fh.write(solo_keep)
    bts2, factss2 = Z.read(SH.solo_adapter("bigcodebench", fx1, seeds))
    assert factss2["n_refusals"] == 0 and factss2["runs"] >= 1


# --------------------------------------------------------------------------------------------
# landing 3 (2026-10-01): the `_l3` sources -- the gemma row's four tuples once more on the updated harness; the three shapes that differ from landing 2
# --------------------------------------------------------------------------------------------

def test_the_l3_specs_bind_the_l3_roots_and_the_producer_key_carries_the_l3_variant_with_ids_disjoint_from_every_earlier_landing():
    """The `_l3` readers are landing 1's readers parametrized as landing 2's are: their own root and source, the same spelling (`<tag>_shared`), a producer key whose SOURCE
    segment carries the variant (`<tag>_shared_l3/<store key>[@rep<N>]`), and landing 3's own account of its shapes (`SharedSpec.landing_rule` = SHARED_L3_RULE; None on
    landings 1 and 2) -- so every run id differs from landing 1's, landing 2's and the solo reader's over one store key while the campaign / eval / instrument prefix is the
    same. Everything else is landing 1's."""
    key, inst = "none__controls__t0__a0__2ff9cdff__1dd7b444", "bigcodebench_full_protocol_greedy_4096_synthetic"
    for l3, l2, l1, solo in ((BCS3, BCS2, BCS, BCF), (BFS3, BFS2, BFS, BFF)):
        s = l3.SPEC
        assert s.source == l3.SOURCE == l3.NAME == f"{solo.SOURCE}_shared_l3" == f"{l1.SOURCE}_l3" and s.root_default.endswith(os.path.join("run-state", "full22", l3.SOURCE))
        assert s.root_default not in (l1.SPEC.root_default, l2.SPEC.root_default, solo.SPEC.root_default) and l3.ROOT_DEFAULT == s.root_default
        assert s.shared is not None and s.shared.harness_variant == "l3" == B.SHARED_L3_HARNESS_VARIANT == l3.HARNESS_VARIANT and s.shared.base_source == l1.SOURCE == l3.BASE_SOURCE
        assert s.shared.landing_rule == B.SHARED_L3_RULE == l3.LANDING_RULE == l3.SHARED_L3_RULE and l2.SPEC.shared.landing_rule is None and l1.SPEC.shared.landing_rule is None
        assert s.shared.solo_source == solo.SOURCE and l3.SOLO_ROOT_DEFAULT == l2.SOLO_ROOT_DEFAULT == l1.SOLO_ROOT_DEFAULT == solo.ROOT_DEFAULT
        for f in ("eval_id", "task", "record_task", "outcomes_file", "instrument_seed", "cap", "protocol_temperature", "item_set_key", "item_set_id", "item_set_label", "item_set_pin_file",
                  "roster_file", "models", "instruct_direct", "clamp_override", "tool_calls_required", "n_items", "three_numbers_file", "launch_set_files", "schedule_file"):
            assert getattr(s, f) == getattr(l1.SPEC, f) == getattr(l2.SPEC, f), f
        assert s.suffix_re.pattern == l1.SPEC.suffix_re.pattern and s.grammar_text == l1.SPEC.grammar_text
        assert l3.spelling_of("gemma4e4b_s1") == l2.spelling_of("gemma4e4b_s1") == l1.spelling_of("gemma4e4b_s1") == "gemma4e4b_s1_shared" and l3.producer_spelling_of("gemma4e4b_s1") == "gemma4e4b_s1_shared_l3"
        assert B.shared_producer_key_for(s, "gemma4e4b_s1", key) == f"gemma4e4b_s1_shared_l3/{key}" and B.shared_producer_key_for(s, "gemma4e4b_s1", key, 2) == f"gemma4e4b_s1_shared_l3/{key}@rep2"
        ids = {"l3": l3.run_id_for("gemma4e4b_s1", key, "none", inst), "l2": l2.run_id_for("gemma4e4b_s1", key, "none", inst), "l1": l1.run_id_for("gemma4e4b_s1", key, "none", inst),
               "solo": solo.run_id_for("gemma4e4b_s1", key, "none", inst), "l3_rep": l3.run_id_for("gemma4e4b_s1", key, "none", inst, repeat=2),
               "l2_rep": l2.run_id_for("gemma4e4b_s1", key, "none", inst, repeat=2), "l1_rep": l1.run_id_for("gemma4e4b_s1", key, "none", inst, repeat=2)}
        assert len(set(ids.values())) == 7, ids
        assert l3.landing2_run_id_for("gemma4e4b_s1", key, "none", inst) == ids["l2"] and l3.landing1_run_id_for("gemma4e4b_s1", key, "none", inst) == ids["l1"]
        assert l3.solo_run_id_for("gemma4e4b_s1", key, "none", inst) == ids["solo"] == l2.solo_run_id_for("gemma4e4b_s1", key, "none", inst)
        parsed = {k: I.parse_run_id(v) for k, v in ids.items()}
        assert len({(p["campaign"], p["eval_id"], p["instrument_prefix"]) for p in parsed.values()}) == 1, "the discriminator is the producer key, never the campaign / eval / instrument"
        assert len({p["producer_sha16"] for p in parsed.values()}) == 7
        assert l3.recount_computation_id() not in (l1.recount_computation_id(), l2.recount_computation_id(), solo.recount_computation_id()) and l3.RECOUNT_ALGORITHM.endswith(f"{l3.NAME}.recount")
        assert l3.SHARED_L2_RUN_ID_RULE == B.SHARED_L2_RUN_ID_RULE and l3.THREE_NUMBERS_SOURCE_RULE == B.THREE_NUMBERS_SOURCE_RULE and l3.SHARED_VERIFIED_STANDUP_RULE == B.SHARED_VERIFIED_STANDUP_RULE


@pytest.mark.parametrize("eval_id", ["bigcodebench", "bfcl"])
def test_l3_stores_in_landing_3s_shape_read_under_the_l3_source_with_ids_disjoint_from_landings_1_and_2_and_the_three_differences_recorded(tmp_path, seeds, eval_id):
    """The gemma row's stores in landing 3's REAL shape (`mirror(landing=3, **LANDING3_SHAPE)`: the fourteen-key verified block built by the real emitter, RESULT's three lines
    with the re-stamp carrying standup= before verified_by=, the legacy route scalar spelling the row's fitted digest on EVERY store) file on the `_l3` source with landing
    1's spellings and contrasts and run ids disjoint from BOTH earlier landings' over the same store keys; the three differences are recorded and none refuses: (1) the
    block is recorded verbatim on the row and on the gate (the reader pins the row_engine block's eight keys, not the verified block's); (2) the standup id is read off the
    LAST line; (3) the scalar is recorded as `object_artifact_sha256_scalar` on the none and prompted rows too and NEVER read as identity -- none files as `none` /
    passthrough, the prompted stores as `prompted_only`, the fitted store as the pinned object, the same object ids landings 1 and 2 file."""
    fx3, ad3, bt3, facts3 = _read(tmp_path / "l3", seeds, eval_id=eval_id, landing=3, **SH.LANDING3_SHAPE)
    fx2, ad2, bt2, facts2 = _read(tmp_path / "l2", seeds, eval_id=eval_id, landing=2, verified=True)
    fx1, ad1, bt1, facts1 = _read(tmp_path / "l1", seeds, eval_id=eval_id)
    assert isinstance(ad3, SH.ADAPTERS_L3[eval_id]) and isinstance(ad2, SH.ADAPTERS_L2[eval_id]) and fx3["keys"] == fx2["keys"] == fx1["keys"]        # the same stores (the same keys)
    assert facts3["n_refusals"] == 0 and facts3["runs"] == 4 == facts2["runs"] == facts1["runs"], facts3["refusals"]
    r3, r2, r1 = SH.runs_by_key(bt3), SH.runs_by_key(bt2), SH.runs_by_key(bt1)
    assert set(r3) == set(r2) == set(r1) == {NONE, RELU, PROMPTED, f"{NONE}@rep2"}
    ids3 = {r["run_id"] for r in r3.values()}
    assert not (ids3 & {r["run_id"] for r in r2.values()}) and not (ids3 & {r["run_id"] for r in r1.values()}), "no landing-3 id equals a landing-2 or landing-1 id"
    l3, l2, l1 = SH.SHARED_L3[eval_id], SH.SHARED_L2[eval_id], SH.SHARED[eval_id]
    # the fixture carries the real landing-3 shapes: RESULT's three lines (the first without a token, the re-stamp with standup= BEFORE verified_by=), the fourteen-key block,
    # the scalar on the none store's manifest AND endpoint while its provenance attests no object
    none_dir = os.path.join(fx3["root"], B.STORES_HEAD, SH.ROW_DIR, NONE, B.THINKING_DIR, f"{fx3['solo_suffix']}-shared")
    with open(os.path.join(none_dir, "RESULT"), encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    assert len(lines) == 3 and "standup=" not in lines[0] and "phase=generate" in lines[0] and f"standup={SH.STANDUP_A}" in lines[1] and "result=GENERATED" in lines[1], lines
    assert "result=COMPLETE status=OK" in lines[-1] and lines[-1].index(f"standup={SH.STANDUP_A}") < lines[-1].index(f"verified_by={SH.VERIFIER_FC}"), lines[-1]
    man = json.load(open(os.path.join(none_dir, "cells", fx3["keys"][NONE], "manifest.json"), encoding="utf-8"))
    geo = json.load(open(os.path.join(none_dir, B.GEOMETRY_FILE), encoding="utf-8"))
    assert man["route"]["object_artifact_sha256"] == Z.OBJ == geo["endpoint"]["object_artifact_sha256"] and not (man.get("provenance") or {}).get("artifact_sha256")
    assert set(man["row_engine"]) == set(B.ROW_ENGINE_KEYS) and set(man["row_engine"]["verification_pass"]) == set(C.VERIFICATION_PASS_VERIFIED_FIELDS) and len(C.VERIFICATION_PASS_VERIFIED_FIELDS) == 14
    for rk, r in r3.items():
        okey, rep = B.split_row_okey(rk)
        ex, re_blk = r["extra"], r["extra"]["row_engine"]
        assert r["lane_source_legacy"] == l3.SOURCE and r["view_legacy"] == r2[rk]["view_legacy"] == r1[rk]["view_legacy"] == f"{TAG}_shared" and r["model_id"] == r2[rk]["model_id"]
        assert r["producer_run_key_legacy"] == f"{TAG}_shared_l3/{fx3['keys'][okey]}{'@rep2' if rep else ''}" and r2[rk]["producer_run_key_legacy"].startswith(f"{TAG}_shared_l2/")
        assert r["run_id"] == l3.run_id_for(TAG, fx3["keys"][okey], okey, fx3["instruments"][eval_id], repeat=rep)
        assert r["run_id"] not in (l2.run_id_for(TAG, fx3["keys"][okey], okey, fx3["instruments"][eval_id], repeat=rep), l1.run_id_for(TAG, fx3["keys"][okey], okey, fx3["instruments"][eval_id], repeat=rep))
        assert ex["solo_run_id"] == r2[rk]["extra"]["solo_run_id"] == fx3["solo_run_ids"][okey][eval_id] and ex["comparison_kind"] == r2[rk]["extra"]["comparison_kind"]
        ss = ex["shared_standup"]
        assert ss["harness_variant"] == "l3" and ss["producer_spelling"] == f"{TAG}_shared_l3" and ss["base_source"] == l1.SOURCE and ss["source_binding"] == "root" and ss["spelling"] == f"{TAG}_shared"
        assert ss["landing_rule"] == B.SHARED_L3_RULE and ss["run_id_rule"] == B.SHARED_L2_RUN_ID_RULE
        assert "landing_rule" not in r2[rk]["extra"]["shared_standup"] and "landing_rule" not in r1[rk]["extra"]["shared_standup"]       # the earlier landings' rows stay byte for byte
        assert ex["launch_set"]["own_source_spelled"] == l1.SOURCE and ex["launch_set"]["own_source_is_base"] is True and ex["source_legacy"] == l3.SOURCE
        # (1) the fourteen-key verified block: recorded verbatim on the row and on the gate, the state and the verifier's call the only fields read
        vp = re_blk["verification_pass"]
        assert tuple(vp) == C.VERIFICATION_PASS_VERIFIED_FIELDS and len(vp) == 14 and vp["state"] == "verified" and vp["function_call_id"] == SH.VERIFIER_FC
        assert vp["verifier_tree_sha"] == vp["generation_tree_sha"] == SH.TREE and vp["binding_version"] == "1" and vp["executed_verify_timeout_s"] == 30 and vp["executed_image"] == SH.SYNTHETIC_IMAGE
        assert ex["gate"]["verification_pass"] == vp and re_blk["verification_state"] == "verified" and re_blk["verification_function_call_id"] == SH.VERIFIER_FC
        assert len(r2[rk]["extra"]["row_engine"]["verification_pass"]) == 9        # landing 2's block beside it, read by the same code
        # (2) the token-present re-stamp: the id read off the LAST line, the row runner's earlier line agreeing; the absent-token branch is not taken
        assert re_blk["result_standup_source"] == "last line" and re_blk["result_standup"] == SH.STANDUP_A == re_blk["result_tokens"]["standup"] == re_blk["standup_id"]
        assert re_blk["result_standup_earlier"] == [SH.STANDUP_A] and re_blk["result_verified_by"] == SH.VERIFIER_FC and re_blk["verified_standup_rule"] == B.SHARED_VERIFIED_STANDUP_RULE
        assert r2[rk]["extra"]["row_engine"]["result_standup_source"].startswith("earlier line")
        assert ex["attempts_before_accepted"] == lines[:2] if rk == NONE else len(ex["attempts_before_accepted"]) == 2
        # (3) the legacy route scalar: the row's fitted digest on EVERY store, recorded, never identity -- none and prompted file as they always did
        assert re_blk["object_artifact_sha256_scalar"] == Z.OBJ and r2[rk]["extra"]["row_engine"]["object_artifact_sha256_scalar"] is None
        if okey == NONE:
            assert r["object_id"] == "none" and r["transfer_kind"] == "none" and r["serving_path"] == "passthrough" and re_blk["served_map_entry"] is None
        elif okey == PROMPTED:
            assert r["object_id"].startswith("si:") and r["transfer_kind"] == "prompted_only" and r["serving_path"] == "passthrough" and re_blk["served_map_entry"] is None
        else:
            assert r["object_id"] == Z.OBJ and r["transfer_kind"] == "base_to_instruct" and r["serving_path"] == "lora_native" and re_blk["served_map_entry"] == Z.OBJ
        assert (r["object_id"], r["transfer_kind"], r["serving_path"]) == (r2[rk]["object_id"], r2[rk]["transfer_kind"], r2[rk]["serving_path"]) == (r1[rk]["object_id"], r1[rk]["transfer_kind"], r1[rk]["serving_path"])
    assert facts3["harness_variant"] == "l3" and facts3["landing_rule"] == B.SHARED_L3_RULE and facts3["three_numbers_source"] == l3.SOURCE and facts3["base_source"] == l1.SOURCE
    assert facts3["launch_set_own_source_spelled"] == l1.SOURCE and facts3["launch_set_own_source_is_base"] is True and facts3["source_binding"] == "root"
    assert "landing_rule" not in facts2 and "landing_rule" not in facts1 and "harness_variant" not in facts1 and facts2["harness_variant"] == "l2"
    # the contrasts are landing 1's: the borrowed comparison against the solo run, the in-run contrast against the row's shared none, the repeat's floor
    assert {c["anchor_kind"] for c in _contrasts(bt3)} == {c["anchor_kind"] for c in _contrasts(bt1)} == {B.ANCHOR_KIND_SOLO_STAND_IN, "same_standup_none"}
    assert len(_contrasts(bt3, B.ANCHOR_KIND_SOLO_STAND_IN)) == 4 and len(_contrasts(bt3, "same_standup_none")) == 3
    assert {c["anchor_run_id"] for c in _contrasts(bt3, B.ANCHOR_KIND_SOLO_STAND_IN)} == set(fx3["solo_run_ids"][k][eval_id] for k in (NONE, RELU, PROMPTED))
    assert set(facts3["standups"]) == {SH.STANDUP_A} and len(facts3["standups"][SH.STANDUP_A]["rows"]) == 4
    # the three-number rows spell object_sha256 None on none / prompted and the digest on the fitted row (the real landing-3 record does): the cross-check agreed, nothing refused
    tn = json.load(open(os.path.join(fx3["root"], B.RECORDS_DIR, l3.THREE_NUMBERS_FILE), encoding="utf-8"))
    assert tn["source"] == l3.SOURCE and tn["rows"][f"{TAG}/{NONE}"].get("object_sha256") is None and tn["rows"][f"{TAG}/{PROMPTED}"].get("object_sha256") is None and tn["rows"][f"{TAG}/{RELU}"].get("object_sha256") == Z.OBJ


def test_the_legacy_route_scalar_is_recorded_and_never_read_as_identity_even_when_it_spells_a_digest_no_store_served(tmp_path, seeds):
    """The code proof behind landing 3's third difference, as a test: the only read of `manifest.route.object_artifact_sha256` in the reader RECORDS it. A scalar spelling a
    digest NO store served -- on the none store, on a prompted store and on the fitted store alike -- changes no object id and refuses nothing: identity is
    `provenance.artifact_sha256` / `fit_provenance.object_sha` plus the by-harness-key map, and the scalar rides along verbatim in `extra.row_engine.object_artifact_sha256_scalar`."""
    fx = SH.mirror(tmp_path, seeds, eval_id="bigcodebench", landing=3, **SH.LANDING3_SHAPE)
    bogus = "f" * 64
    for okey in (NONE, PROMPTED, RELU):
        d = os.path.join(fx["root"], B.STORES_HEAD, SH.ROW_DIR, okey, B.THINKING_DIR, f"{fx['solo_suffix']}-shared")
        mp = os.path.join(d, "cells", fx["keys"][okey], "manifest.json")
        man = json.load(open(mp, encoding="utf-8"))
        man["route"]["object_artifact_sha256"] = bogus
        X.write(mp, man)
        gp = os.path.join(d, B.GEOMETRY_FILE)
        geo = json.load(open(gp, encoding="utf-8"))
        geo["endpoint"]["object_artifact_sha256"] = bogus
        X.write(gp, geo)
        X.write(os.path.join(d, "standup", "endpoint.json"), dict(geo["endpoint"]))
    out, _ = SH.read_all(SH.adapter("bigcodebench", fx, seeds))
    bt, facts = out[GROUP]
    assert facts["n_refusals"] == 0 and facts["runs"] == 4, facts["refusals"]
    runs = SH.runs_by_key(bt)
    assert runs[NONE]["object_id"] == "none" and runs[PROMPTED]["object_id"].startswith("si:") and runs[RELU]["object_id"] == Z.OBJ and runs[RELU]["extra"]["row_engine"]["served_map_entry"] == Z.OBJ
    assert {runs[k]["extra"]["row_engine"]["object_artifact_sha256_scalar"] for k in (NONE, PROMPTED, RELU)} == {bogus} and runs[f"{NONE}@rep2"]["extra"]["row_engine"]["object_artifact_sha256_scalar"] == Z.OBJ
    assert {runs[k]["extra"]["row_engine"]["object_artifact_sha256_by_harness_key"][X.harness_id(RELU)] for k in runs} == {Z.OBJ}, "the map, not the scalar, is the served digest"


def test_the_reader_pins_neither_verified_key_set_the_nine_key_block_reads_on_the_l3_reader_and_the_fourteen_key_block_on_the_l2_and_l1_readers(tmp_path, seeds):
    """Landing 3's first difference is not a gate anywhere: the registry reader pins the row_engine block's EIGHT keys and reads the verified block's `state` and
    `function_call_id` alone (it imports nothing from capeval.tinycap.contracts), so landing 2's nine-key block reads on the `_l3` reader, landing 3's fourteen-key block
    reads on the `_l2` and landing-1 readers (the rules are the store's, not the source's), and a fourteen-key block whose state is not `verified` with no token on the
    last line still refuses under the landing-1 rule -- the STATE decides, never the key count. The fixture's fourteen-key block is the real emitter's output
    (`cpu_verify.verified_block`), so the test proves the reader against the emitter's shape and not a copy of it."""
    fx, ad, bt, facts = _read(tmp_path / "a", seeds, landing=3, verified=True)                      # landing 2's shapes on the l3 reader
    assert isinstance(ad, BCS3.BigcodebenchFullSharedL3Adapter) and facts["n_refusals"] == 0 and facts["runs"] == 4, facts["refusals"]
    for r in bt["run"]:
        re_blk = r["extra"]["row_engine"]
        assert len(re_blk["verification_pass"]) == 9 and set(re_blk["verification_pass"]) == set(C.VERIFICATION_PASS_VERIFIED_FIELDS_V1)
        assert re_blk["result_standup_source"].startswith("earlier line") and re_blk["object_artifact_sha256_scalar"] is None and r["extra"]["shared_standup"]["landing_rule"] == B.SHARED_L3_RULE
    fx2, ad2, bt2, facts2 = _read(tmp_path / "b", seeds, landing=2, **SH.LANDING3_SHAPE)              # landing 3's shapes on the l2 reader
    assert isinstance(ad2, BCS2.BigcodebenchFullSharedL2Adapter) and facts2["n_refusals"] == 0 and facts2["runs"] == 4, facts2["refusals"]
    for r in bt2["run"]:
        re_blk = r["extra"]["row_engine"]
        assert len(re_blk["verification_pass"]) == 14 and re_blk["result_standup_source"] == "last line" and re_blk["object_artifact_sha256_scalar"] == Z.OBJ
        assert "landing_rule" not in r["extra"]["shared_standup"] and r["extra"]["shared_standup"]["harness_variant"] == "l2"
    fx1, ad1, bt1, facts1 = _read(tmp_path / "c", seeds, landing=1, **SH.LANDING3_SHAPE)              # and on landing 1's reader
    assert isinstance(ad1, BCS.BigcodebenchFullSharedAdapter) and facts1["n_refusals"] == 0 and facts1["runs"] == 4
    assert all(r["extra"]["row_engine"]["result_standup_source"] == "last line" and len(r["extra"]["row_engine"]["verification_pass"]) == 14 for r in bt1["run"])
    assert all("harness_variant" not in r["extra"]["shared_standup"] for r in bt1["run"])
    # fourteen keys, state pending, no token on the last line: the landing-1 rule refuses every store (the state decides)
    fx4, ad4, bt4, facts4 = _read(tmp_path / "d", seeds, landing=3, verified=True, verified_fields="current", verified_state_override="pending")
    assert facts4["runs"] == 0 and len(facts4["refusals"]) == 4 and all("RESULT's last line carries standup='<none>'" in x and B.SHARED_STANDUP_RULE[:30] in x for x in facts4["refusals"])
    # the fixture's blocks ARE the contract's two sets; the fourteen-key one is the emitter's own output
    cur, v1 = SH.verification_block(fields="current"), SH.verification_block()
    assert tuple(cur) == C.VERIFICATION_PASS_VERIFIED_FIELDS and set(v1) == set(C.VERIFICATION_PASS_VERIFIED_FIELDS_V1) and len(cur) == 14 and len(v1) == 9
    assert C.verified_block_keys_problem(cur) is None and C.verified_block_keys_problem(v1) is None and C.verified_block_keys_problem(dict(cur, extra_key=1)) is not None
    assert cur["sandbox_facts"]["binding_version"] == "1" == cur["binding_version"]       # the real landing-3 block's sandbox_facts carry binding_version too


def test_the_l3_readers_are_registered_declared_and_versioned():
    from sisteer.registry import adapters as ADP
    reg = ADP.load_all()
    for mod, cls, l2, l1 in ((BCS3, BCS3.BigcodebenchFullSharedL3Adapter, BCS2, BCS), (BFS3, BFS3.BfclFullSharedL3Adapter, BFS2, BFS)):
        assert mod.NAME in reg and isinstance(reg[mod.NAME](), cls) and cls.covers_sources == (mod.SOURCE,) and AB.owner_of(cls()) == "Overflow" == mod.OWNER
        assert AB.emission_version_of(cls) == "3" and cls.SPEC.shared is not None and cls.SPEC.shared.harness_variant == "l3" and mod.BASE_SOURCE == l1.SOURCE   # 3: solo_stand_in (migration 0027)
        assert cls(root="/nonexistent/x").status_sources("gemma4e4b_s1") == (mod.SOURCE,) and cls(root="/nonexistent/x").solo_root == l1.SOLO_ROOT_DEFAULT
        cls2 = l2.BigcodebenchFullSharedL2Adapter if l2 is BCS2 else l2.BfclFullSharedL2Adapter
        cls1 = l1.BigcodebenchFullSharedAdapter if l1 is BCS else l1.BfclFullSharedAdapter
        assert mod.SOURCE not in cls2.covers_sources and mod.SOURCE not in cls1.covers_sources and l2.SOURCE not in cls.covers_sources and l1.SOURCE not in cls.covers_sources
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "registry", "sources_r2_v1.yaml"), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    names = {r["source"]: r["owner"] for r in data["sources"]}
    adds = {a["source"]: a for a in data["meta"]["additions"]}
    assert data["meta"]["n_sources"] == len(data["sources"]) == len(names)
    for s in (BCS3.SOURCE, BFS3.SOURCE):
        assert names[s] == "the authors" and adds[s]["provisional"] is True and adds[s]["paper_build_commit"] is None and "COMPARISON" in adds[s]["why"]
        assert "collide" in adds[s]["why"] or "landing 1" in adds[s]["why"].lower()
        assert "registry" in adds[s]["declared_by"] and adds[s]["producer_message_utc"] == "2026-10-01T02:28Z"
    assert "never read as identity" in adds[BCS3.SOURCE]["why"] and "fourteen" in adds[BCS3.SOURCE]["why"] and "standup=<manifest id> before verified_by=" in adds[BCS3.SOURCE]["why"]


def test_a_three_numbers_record_of_another_landing_under_the_l3_root_refuses_every_row_by_name_in_both_directions(tmp_path, seeds):
    """THREE_NUMBERS_SOURCE_RULE across three landings: the SAME file name sits under every landing's root, so a landing-2 record under the `_l3` root, a landing-3 record
    under the `_l2` root and a landing-3 record under the landing-1 root each refuse every row by name (the file, the field `source`, both names), filing nothing under a
    wrong landing's ids; the real-shaped fixtures still read once restored. (The `_l2` reader over the REAL landing-3 root refused exactly this way on 2026-10-01: 4
    refusals, 0 runs.)"""
    l1, l2, l3 = SH.SHARED["bigcodebench"], SH.SHARED_L2["bigcodebench"], SH.SHARED_L3["bigcodebench"]
    fx3 = SH.mirror(tmp_path / "l3", seeds, eval_id="bigcodebench", landing=3, **SH.LANDING3_SHAPE)
    fx2 = SH.mirror(tmp_path / "l2", seeds, eval_id="bigcodebench", landing=2, verified=True)
    fx1 = SH.mirror(tmp_path / "l1", seeds, eval_id="bigcodebench")
    paths = {n: os.path.join(fx["root"], B.RECORDS_DIR, l3.THREE_NUMBERS_FILE) for n, fx in (("l1", fx1), ("l2", fx2), ("l3", fx3))}
    assert l1.THREE_NUMBERS_FILE == l2.THREE_NUMBERS_FILE == l3.THREE_NUMBERS_FILE
    keep = {n: open(p, "rb").read() for n, p in paths.items()}
    assert json.loads(keep["l3"])["source"] == l3.SOURCE and json.loads(keep["l2"])["source"] == l2.SOURCE
    for root_name, record_name, fx, reader_mod, other_mod in (("l3", "l2", fx3, l3, l2), ("l2", "l3", fx2, l2, l3), ("l1", "l3", fx1, l1, l3)):
        with open(paths[root_name], "wb") as fh:
            fh.write(keep[record_name])
        out, _ = SH.read_all(SH.adapter("bigcodebench", fx, seeds))
        bt, facts = out[GROUP]
        assert facts["runs"] == 0 and not bt.get("run") and len(facts["refusals"]) == 4, (root_name, facts["refusals"])
        why = _refusal(facts, f"{TAG}|{NONE}")
        assert f"{reader_mod.THREE_NUMBERS_FILE}: the three-numbers record spells source '{other_mod.SOURCE}'; this reader is '{reader_mod.SOURCE}' -- wrong root or wrong mirror" in why
        assert B.THREE_NUMBERS_SOURCE_RULE[:40] in why
        if root_name != "l1":
            assert facts["three_numbers_source"] == other_mod.SOURCE
        else:
            assert "three_numbers_source" not in facts        # landing 1's status stays byte for byte
        with open(paths[root_name], "wb") as fh:
            fh.write(keep[root_name])
    for fx, src in ((fx3, l3.SOURCE), (fx2, l2.SOURCE), (fx1, l1.SOURCE)):
        out, _ = SH.read_all(SH.adapter("bigcodebench", fx, seeds))
        assert out[GROUP][1]["runs"] == 4 and out[GROUP][1]["n_refusals"] == 0 and out[GROUP][1].get("three_numbers_source", src) == src
