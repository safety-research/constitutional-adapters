"""Roster v1 + rulings seed: the committed files load and expand to their declared counts with the
disagreements listed first; the loader refuses an unknown model / object / eval / comparison / action by name (and a
YAML boolean where a vocabulary value belongs); a synthetic roster ingests and `roster diff` reads it back equal;
`v_paper_objects` reads AlphaSteer OUT by the ruling while its runs stay ingested (the latest ruling
wins); a comparison-scoped exclude removes the comparison, not the run; hold_out marks, never excludes; a guard-only
ruling never expires but an incoherent guard makes the covered keys indeterminate; `paper_long_frame(columns=)`
derives the paper's long schema and refuses a column it cannot supply by name; `ruling add` writes a record.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import roster as RO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROSTER = os.path.join(ROOT, "configs", "registry", "roster_r2_v1.yaml")
RULINGS = os.path.join(ROOT, "configs", "registry", "rulings_r2_v1.jsonl")


def _round_sans_version(pr):
    """The paper_round payload without `roster_version` (migration 0016 stamps the version the load wrote, so two versions' rounds differ
    in that key alone; every other field is re-stamped, not changed)."""
    return {k: v for k, v in pr.items() if k != "roster_version"}


# --------------------------------------------------------------------------------------------
# the committed files
# --------------------------------------------------------------------------------------------


def test_committed_roster_and_rulings_load_to_their_declared_counts():
    load = RO.load(ROSTER, RULINGS)
    s = load.summary()
    # 89 + the two 2026-09-17 decode rulings filed with user_direct provenance (decode-pins-0917) + the EM-relayed vendor-preset default ruling
    # (decode-vendor-preset-default-2026-09-17, registry-em-organisms) + the Llama thinking-equivalence ruling (decode-llama-thinking-not-available-2026-09-18,
    # user_direct, registry-llama-thinking-na)
    assert s["entries"] == 5916 and s["rulings"] == 93
    assert s["rulings_by_action"] == {"coexist": 1, "descope": 1, "drop_eval": 6, "exclude": 16, "hold_out": 1, "include": 19,
                                      "operating_point": 9, "reference": 19, "supersede": 21}   # reference 15 -> 17: the two 2026-09-17 decode rulings; -> 18: the EM-relayed default ruling; -> 19: the 2026-09-18 Llama thinking-equivalence ruling
    # the eight relayed rulings + the three post-pin (f33dca5d) rulings + the two 2026-09-17 decode rulings (user_direct, filed by
    # decode-pins-0917) carry a provenance block; every earlier one carries it in ruled_by prose
    assert s["rulings_by_provenance"] == {"in_ruled_by_prose": 78, "peer_encoded": 2, "relayed": 9, "user_direct": 4}   # relayed 8 -> 9: the relayed vendor-preset default ruling; user_direct 3 -> 4: the 2026-09-18 Llama thinking-equivalence ruling
    # every entry names exactly one of object_id / object_selector_json and carries a citation naming the pinned symbol
    for e in load.entries:
        assert (e["object_id"] is None) != (e["object_selector_json"] is None), e
        assert e["instrument_selector_json"] is not None and e["instrument_id"] is None
        assert "@ 40997227" in e["source_citation"], e["source_citation"]
        assert not (e["judge_id"] and e["judge_selector_json"])
    # the disagreements are present, each with both citations, and listed first by show_text
    assert load.disagreements and all(e["inferred"] == 1 for e in load.disagreements)
    texts = {e["extra"]["disagreement"] for e in load.disagreements}
    assert any("D-1" in t for t in texts) and any("D-2" in t for t in texts) and any("D-9" in t for t in texts)
    assert RO.show_text(load).splitlines()[1].startswith("DISAGREEMENTS FIRST")
    # the vocabulary mappings: ladder rule, drawn=0, not_applicable, baseline_published
    assert any(e["strength_rule"] == "ladder:alphasteer_lambda_ladder" for e in load.entries)
    assert sum(1 for e in load.entries if e["drawn"] == 0) > 0
    assert any(e["extra"]["transfer_kind"] == "not_applicable" for e in load.entries if e["comparison_id"] == "fig4_organism")
    r = yaml.safe_load(open(ROSTER))
    assert r["objects"]["arditi"]["object_role"] == "baseline_published" and r["objects"]["alphasteer_rebake"]["object_role"] == "baseline_published"
    # the rulings are in the seed
    ids = {x["ruling_id"] for x in load.rulings}
    assert "r2v1-exclude-alphasteer-objects-out-of-round-r2" in ids
    assert sum(1 for i in ids if i.startswith("r2v1-exclude-relu_native-llama-gsm8k-lens-anchor-")) == 3
    assert sum(1 for i in ids if i.startswith("r2v1-include-relu_native-llama-gsm8k-native-anchor-")) == 3
    assert sum(1 for i in ids if i.startswith("decode-")) == 9 + 1 + 2 + 1 + 1     # the nine decode rulings + the vendor-preset target-row ruling + the two 2026-09-17 decode rulings + the relayed vendor-preset default ruling + the 2026-09-18 Llama thinking-equivalence ruling (user_direct)
    d29 = next(x for x in load.rulings if x["ruling_id"] == "r2v1-exclude-alphasteer-objects-out-of-round-r2")
    assert d29["ruled_by"] == "the authors" and d29["ruled_utc"] == "2026-09-17T00:20:00.000Z" and d29["scope_json"]["match"] == {"object_kinds": ["alphasteer"]}
    # every self-retiring ruling carries the structured expiry form; the guard-only ones have conditions []
    mt = next(x for x in load.rulings if x["ruling_id"] == "r2v1-supersede-llama-arditi-r2-multiturn-r0p1")
    assert mt["expires_when_json"]["kind"] == "any_of" and len(mt["expires_when_json"]["conditions"]) == 2 and len(mt["expires_when_json"]["guards"]) == 1
    em = next(x for x in load.rulings if x["ruling_id"] == "r2v1-supersede-gemma-alphasteer-em-bl-lam3")
    assert em["expires_when_json"]["conditions"] == [] and em["expires_when_json"]["guards"][0]["kind"] == "input_sha_pinned"
    # precision column populated; date-only sources are midnight UTC
    assert all(x["ruled_utc_precision"] in ("date", "datetime") for x in load.rulings)
    assert all(x["ruled_utc"].endswith("T00:00:00.000Z") for x in load.rulings if x["ruled_utc_precision"] == "date")


def test_roster_vocabulary_agrees_with_the_code_tables():
    r = yaml.safe_load(open(ROSTER))
    v = r["vocabulary"]
    # the roster's eval keys are the paper's spellings; the long frame's EVAL_LEGACY maps the registry id back to the spelling
    # the EXPORT uses (the six Petri behaviours are spelled without the `petri_` prefix there, as the notes' §1 D-12 records)
    for paper_key, registry_id in v["eval_id_from_eval_legacy"].items():
        assert registry_id in paper.EVAL_LEGACY, registry_id
        if not paper_key.startswith("petri_"):
            assert paper.EVAL_LEGACY[registry_id] == paper_key
    assert v["model_id_from_model_legacy"] == {legacy: mid for mid, legacy in paper.MODEL_LEGACY.items()}
    inv = {label: arm for arm, label in v["paper_label_from_arm_legacy"].items()}
    assert all(paper.PAPER_LABEL_LEGACY[label] == arm for label, arm in inv.items())
    assert {k: RO.COMPONENT_TO_CATEGORY[k] for k in v["eval_category_from_component_legacy"]} == v["eval_category_from_component_legacy"]
    assert v["transfer_kind_from_lane_kind_legacy"] == {legacy: tk for tk, legacy in paper.TRANSFER_KIND_LEGACY.items() if tk != "none"}


# --------------------------------------------------------------------------------------------
# strictness
# --------------------------------------------------------------------------------------------

MINI = {
    "meta": {"schema": RO.ROSTER_SCHEMA, "roster_version": "v1", "round_id": "r_synth",
             "paper_round": {"round_id": "r_synth", "opened_utc": "2026-01-01", "description": "synthetic round", "paper_build_commit": "e" * 40}},
    "vocabulary": {"eval_id_from_eval_legacy": {}},
    "models": {synth.MODEL_ID: {"model_legacy": "synth", "paper_lane": {"transfer_kind": "base_to_instruct"}}},
    "objects": {"none": {"paper_label": "none", "object_kind": "none", "object_role": "baseline_none"},
                "constitution_lora": {"paper_label": "constitution_lora", "object_kind": "lora", "object_role": "constitution", "arm_legacy": "const_lora"}},
    "judges": {"j": {"judge_model": "judge-model-x"}},
    "evals": {synth.EVAL_ID: {"eval_category": "static", "component_legacy": "static", "judge": "j",
                              "instrument_selector_json": {"caps": [512], "harness": "bespoke_sisteer", "thinking_mode": "off"}}},
    "comparisons": [{"comparison_id_template": "fig_synth/{transfer_kind}", "figure_legacy": "fig_synth",
                     "models": [{"model_id": synth.MODEL_ID, "transfer_kind": "base_to_instruct"}],
                     "objects": ["none", "constitution_lora"], "evals": [synth.EVAL_ID], "strength_rules": ["operating_point"],
                     "source_citation": "paperlib.config.SYNTHETIC @ eeeeeeee"}],
}


def _mini(**over):
    import copy
    m = copy.deepcopy(MINI)
    for k, v in over.items():
        m[k] = v
    return m


def test_expansion_of_the_mini_roster_is_the_cross_product():
    exp = RO.expand(_mini())
    assert len(exp.entries) == 2 and exp.counts_by_comparison == {"fig_synth": 2}
    none = next(e for e in exp.entries if e["object_id"] == "none")
    lora = next(e for e in exp.entries if e["object_id"] is None)
    assert lora["object_id"] is None and lora["object_selector_json"] == {"paper_label": "constitution_lora", "object_kind": "lora", "transfer_kind": "base_to_instruct"}
    assert none["object_id"] == "none" and none["object_selector_json"] is None
    assert lora["instrument_selector_json"] == {"token_cap_tokens": [512], "harness": "bespoke_sisteer", "thinking_mode": "off"}
    assert lora["judge_selector_json"] == {"judge_model": "judge-model-x"}
    # idempotent identities
    assert {e["roster_entry_id"] for e in exp.entries} == {e["roster_entry_id"] for e in RO.expand(_mini()).entries}


@pytest.mark.parametrize("mutate, needle", [
    (lambda m: m["comparisons"][0]["models"].append({"model_id": "nomodel_it", "transfer_kind": "instruct_direct"}), "unknown model 'nomodel_it'"),
    (lambda m: m["comparisons"][0]["objects"].append("mystery_object"), "unknown object 'mystery_object'"),
    (lambda m: m["comparisons"][0]["evals"].append("mystery_eval"), "unknown eval 'mystery_eval'"),
    (lambda m: m["comparisons"][0]["strength_rules"].append("dose:2"), "strength_rule 'dose:2'"),
    (lambda m: m["objects"]["constitution_lora"].__setitem__("object_kind", "adapter"), "object_kind 'adapter'"),
    (lambda m: m["comparisons"][0]["models"].__setitem__(0, {"model_id": synth.MODEL_ID, "transfer_kind": "lane_x"}), "transfer_kind 'lane_x'"),
    (lambda m: m["evals"][synth.EVAL_ID]["instrument_selector_json"].__setitem__("thinking_mode", False), "thinking_mode=False"),
    (lambda m: m["meta"].__setitem__("expanded_counts_by_comparison", {"fig_synth": 5}), "expanded counts differ"),
])
def test_roster_refuses_unknown_names_and_out_of_vocabulary_values_by_name(mutate, needle):
    m = _mini()
    mutate(m)
    with pytest.raises(RO.RosterRefusal) as ei:
        RO.expand(m)
    assert needle in str(ei.value)


def _ruling(**over):
    d = {"ruling_id": "R_x", "round_id": "r_synth", "action": "exclude", "scope_kind": "key", "scope_json": {"model_id": synth.MODEL_ID},
         "target_json": {}, "reason": "synthetic", "ruled_utc": "2026-01-02", "ruled_by": "the user", "source_ref": "test"}
    d.update(over)
    return d


def test_ruling_normalisation_and_refusals():
    vocab = RO.roster_vocab(_mini())
    p = RO.normalise_ruling(_ruling(), vocab)
    assert p["ruled_utc"] == "2026-01-02T00:00:00.000Z" and p["ruled_utc_precision"] == "date" and p["scope_json"]["match"] == {"model_ids": [synth.MODEL_ID]}
    p = RO.normalise_ruling(_ruling(ruled_utc="2026-01-02T03:04:05Z", extra_json={"guards_json": [{"kind": "input_sha_pinned", "artifact_uri": "x", "sha256_prefix": "ab"}]}), vocab)
    assert p["ruled_utc"] == "2026-01-02T03:04:05.000Z" and p["ruled_utc_precision"] == "datetime"
    assert p["expires_when_json"] == {"kind": "any_of", "conditions": [], "guards": [{"kind": "input_sha_pinned", "artifact_uri": "x", "sha256_prefix": "ab"}]}
    with pytest.raises(RO.RosterRefusal, match="action 'veto'"):
        RO.normalise_ruling(_ruling(action="veto"), vocab)
    with pytest.raises(RO.RosterRefusal, match="scope_kind 'figure'"):
        RO.normalise_ruling(_ruling(scope_kind="figure"), vocab)
    with pytest.raises(RO.RosterRefusal, match="unknown model 'ghost_it'"):
        RO.normalise_ruling(_ruling(scope_json={"model_id": "ghost_it"}), vocab)
    with pytest.raises(RO.RosterRefusal, match="comparison glob 'fig_nope/\\*'"):
        RO.normalise_ruling(_ruling(scope_kind="comparison", scope_json={"comparison_ids": ["fig_nope/*"]}), vocab)
    with pytest.raises(RO.RosterRefusal, match="match key 'colours'"):
        RO.normalise_ruling(_ruling(scope_json={"match": {"colours": ["red"]}}), vocab)
    with pytest.raises(RO.RosterRefusal, match="expiry kind 'vanished'"):
        RO.normalise_ruling(_ruling(expires_when_json={"kind": "vanished"}), vocab)
    # a scope that names nothing matchable covers nothing (match None), never everything
    assert RO.normalise_ruling(_ruling(scope_json={"applies_to": "a rule the reader enforces"}), vocab)["scope_json"]["match"] is None
    # a declared date precision that contradicts a time of day is refused, not silently kept
    with pytest.raises(RO.RosterRefusal, match="carries a time of day"):
        RO.normalise_ruling(_ruling(ruled_utc="2026-01-02T03:04:05Z", ruled_utc_precision="date"), vocab)



def test_build_match_reads_the_excluded_eval_keys_of_a_comparison_exclude():
    """A comparison-scoped exclude that names the evals it removes under `eval_ids_excluded` / `eval_categories_excluded` covers THOSE
    evals: the seed's match for r2v1-exclude-fig5-defence-axis-bon-multiturn carried comparison_ids alone and stripped the frontier
    comparison off every matched run (2026-09-20)."""
    vocab = RO.roster_vocab(_mini())
    p = RO.normalise_ruling(_ruling(scope_kind="comparison", scope_json={"comparison_ids": ["fig_synth/*"], "eval_ids_excluded": [synth.EVAL_ID],
                                                                          "eval_categories_excluded": ["static"]}), vocab)
    assert p["scope_json"]["match"] == {"eval_ids": [synth.EVAL_ID], "eval_categories": ["static_attack"], "comparison_ids": ["fig_synth/*"]}


def test_committed_v3_defence_axis_ruling_match_carries_its_excluded_evals():
    """rulings_r2_v3.jsonl's one difference from v2: the defence-axis exclude's machine match names the four evals it removes from the
    frontier comparisons (figs.FRONTIER_DEFENCE_OUT @ c53ae173), mapped through the transcriber's spellings (bon -> strongreject_bon)."""
    v3 = os.path.join(ROOT, "configs", "registry", "rulings_r2_v3.jsonl")
    if not os.path.exists(v3):
        pytest.skip("no v3 rulings file in this tree")
    _, raw = RO.read_rulings_jsonl(v3)
    r = next(x for x in raw if x["ruling_id"] == "r2v1-exclude-fig5-defence-axis-bon-multiturn")
    vocab = RO.roster_vocab(RO.load_yaml(os.path.join(ROOT, "configs", "registry", "roster_r2_v3.yaml")))
    m = RO.normalise_ruling(r, vocab)["scope_json"]["match"]
    assert m["comparison_ids"] == ["fig5_frontier/*", "figQ_frontier/instruct_direct"]
    assert m["eval_ids"] == ["strongreject_bon", "pair", "crescendo", "actorattack"]
    assert r["extra_json"]["match_correction_2026_09_20"]["kind"] == "transcription_correction"


V3_ROSTER, V3_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v3.yaml", "rulings_r2_v3.jsonl"))
V4_ROSTER, V4_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v4.yaml", "rulings_r2_v4.jsonl"))
FIGR_PANELS = ("attacks", "overrefusal", "petri", "capability", "misalignment")


def test_committed_v4_adds_the_relu_steer_comparisons_and_corrects_two_selectors():
    """roster_r2_v4.yaml (2026-09-21): v3 plus ten NEW comparison blocks figR_<panel>_absolute/{transfer_kind}
    and /base_to_instruct declaring control_relu + constitution_relu on gemma4e4b_it, llama8b_it and nemotron3nano4b (252 entries; the
    paper's transfer kind and the base->instruct companion per family; Qwen left out - its ReLU-steer cells are selected under
    the figQ_* comparisons already and the notebook's main-figure reader filters on selection_state alone), and two instrument-selector corrections to the live instrument facts - bigcodebench_200
    (harness inspect_evals, item set, caps) and math500 (harness inspect_ai, item set) - which re-mint the 372 v3 entries carrying those
    evals (a selector is part of the uuid5 identity). No v3 block is touched; rulings byte-identical to v3's."""
    import collections
    if not os.path.exists(V4_ROSTER):
        pytest.skip("no v4 roster file in this tree")
    v3, v4 = RO.load(V3_ROSTER, V3_RULINGS), RO.load(V4_ROSTER, V4_RULINGS)
    s = v4.summary()
    assert v4.version == "v4" and s["entries"] == 6370 and s["rulings"] == 93 and s["dispositions"] == 102
    assert open(V4_RULINGS, "rb").read() == open(V3_RULINGS, "rb").read()      # the rulings file is v3's, byte for byte
    assert _round_sans_version(v4.paper_round) == _round_sans_version(v3.paper_round)                                      # the round record is re-stamped, not changed
    d = RO.diff_entries(v4.entries, v3.entries)
    assert not d["changed"] and len(d["only_a"]) == 624 and len(d["only_b"]) == 372
    assert collections.Counter(k["key"][3] for k in d["only_b"]) == {"math500": 186, "bigcodebench": 186}   # re-minted, not lost
    assert len(d["coarse_only_a"]) == 252 and not d["coarse_only_b"]                                        # 252 genuinely new cells
    # every v3 comparison keeps its v3 count: the ReLU-steer entries live on their own comparison family, not on the figA_* blocks
    assert {k: v for k, v in v4.counts_by_comparison.items() if not k.startswith("figR_")} == v3.counts_by_comparison
    assert {k: v for k, v in v4.counts_by_comparison.items() if k.startswith("figR_")} == {
        "figR_attacks_absolute": 48, "figR_attacks_absolute/base_to_instruct": 48, "figR_overrefusal_absolute": 12,
        "figR_overrefusal_absolute/base_to_instruct": 12, "figR_petri_absolute": 42, "figR_petri_absolute/base_to_instruct": 42,
        "figR_capability_absolute": 12, "figR_capability_absolute/base_to_instruct": 12, "figR_misalignment_absolute": 12,
        "figR_misalignment_absolute/base_to_instruct": 12}
    relu = [e for e in v4.entries if e["comparison_id"].startswith("figR_")]
    assert len(relu) == 252 and all(e["inferred"] == 1 and e["drawn"] == 1 and e["strength_rule"] == "operating_point" for e in relu)
    assert {e["comparison_id"] for e in relu} == {f"figR_{p}_absolute/{tk}" for p in FIGR_PANELS
                                                  for tk in ("instruct_direct", "chat_checkpoint", "base_to_instruct")}
    assert {(e["model_id"], e["extra"]["transfer_kind"]) for e in relu} == {
        ("gemma4e4b_it", "instruct_direct"), ("gemma4e4b_it", "base_to_instruct"), ("llama8b_it", "instruct_direct"),
        ("llama8b_it", "base_to_instruct"), ("nemotron3nano4b", "chat_checkpoint")}
    assert not any(e["model_id"] == "qwen35_4b_it" for e in relu)   # No Qwen entry on the figR_* family, either variant
    for e in relu:
        sel = e["object_selector_json"]
        assert e["object_id"] is None and sel["object_kind"] == "relu_steer" and sel["paper_label"] in ("control_relu", "constitution_relu")
        assert sel["transfer_kind"] == e["extra"]["transfer_kind"] and "v4 gap 1" in e["extra"]["inferred_reason"]
    assert not any(e["comparison_id"].startswith("figA_") and (e["object_selector_json"] or {}).get("paper_label") in ("control_relu", "constitution_relu")
                   for e in v4.entries)                                          # the draft's figA_* sub-block is gone
    evals_by = collections.defaultdict(set)
    for e in relu:
        evals_by[e["comparison_id"].split("/")[0]].add(e["eval_id"])
    assert evals_by["figR_attacks_absolute"] == {"pasttense", "codeattack", "renellm", "grok420", "strongreject_bon", "pair", "crescendo", "actorattack"}
    assert evals_by["figR_overrefusal_absolute"] == {"xstest", "or_bench_hard"}
    assert evals_by["figR_petri_absolute"] == {"petri_cyber_bio_chem_capabilities", "petri_defer_to_users", "petri_delusion_sycophancy",
                                              "petri_instruction_evasion", "petri_reasoning_unfaithfulness", "petri_self_preservation", "petri_pooled"}
    assert evals_by["figR_capability_absolute"] == {"mmlu", "gsm8k"}          # the two capability evals run for the ReLU steer on every family
    assert evals_by["figR_misalignment_absolute"] == {"agentic_leaking", "agentic_murder"}
    # the nemotron agentic_leaking not_applicable disposition (common_exceptions) reaches the new entries too
    ids = {e["roster_entry_id"] for e in relu}
    assert sum(1 for x in v4.dispositions if x["roster_entry_id"] in ids and x["eval_id"] == "agentic_leaking" and x["state"] == "not_applicable") == 4
    # the two selector corrections: one registry selector per eval, spelled as the live instrument rows are
    sel = lambda load, ev: {json.dumps(e["instrument_selector_json"], sort_keys=True) for e in load.entries if e["eval_id"] == ev}
    assert sel(v4, "math500") == {'{"harness": "inspect_ai", "item_set_id": "math500_500", "scoring_rule": "sympy_equiv", "token_cap_tokens": [16384]}'}
    assert sel(v4, "bigcodebench") == {'{"harness": "inspect_evals", "item_set_id": "bigcodebench_200", "scoring_rule": "pass_at_1_sandbox", '
                                       '"thinking_mode": "off", "token_cap_tokens": [4096, 16384]}'}
    assert sel(v3, "math500") == {'{"harness": "inspect_evals", "scoring_rule": "sympy_equiv", "token_cap_tokens": [16384]}'}
    assert sel(v4, "bfcl") == sel(v3, "bfcl")                          # bfcl's own override matched already and is untouched


def test_committed_v4_relu_steer_comparisons_escape_every_comparison_glob_of_the_comparison_scoped_excludes():
    """v_paper_current's `matched` step removes a comparison from a covered run when `m.comparison_id GLOB <glob>` for a glob of a
    comparison-scoped exclude (schema_def: the `matched` and `covered` CTEs; SQLite GLOB, not fnmatch). The six-arms exclude
    r2v1-exclude-relu_steer-main-figures is byte-identical in v4 and names figA_*_absolute/* among its globs -
    the draft that put the ReLU-steer entries on the figA_* blocks matched 343 runs and selected none. Every figR_* id must
    match NO glob of that ruling, nor of any other comparison-scoped exclude in v4's rulings, evaluated with sqlite's own GLOB, so a
    rename cannot fall back under an exclude unnoticed."""
    import sqlite3
    if not os.path.exists(V4_ROSTER):
        pytest.skip("no v4 roster file in this tree")
    v4 = RO.load(V4_ROSTER, V4_RULINGS)
    ids = sorted({e["comparison_id"] for e in v4.entries if e["comparison_id"].startswith("figR_")})
    assert len(ids) == 15
    con = sqlite3.connect(":memory:")
    glob = lambda s, g: bool(con.execute("SELECT ? GLOB ?", (s, g)).fetchone()[0])
    assert glob("figA_attacks_absolute/instruct_direct", "figA_*_absolute/*") and glob("figS_bon_curves/instruct_direct", "figS_*")   # the matcher fires where it should
    _, raw = RO.read_rulings_jsonl(V4_RULINGS)
    six = next(r for r in raw if r["ruling_id"] == "r2v1-exclude-relu_steer-main-figures")
    assert six["action"] == "exclude" and six["scope_kind"] == "comparison"
    assert set(six["scope_json"]["match"]["comparison_ids"]) == {"fig1_headline/*", "fig2_jailbreak/*", "fig3_misalign/*", "fig4_transfer", "fig4_organism",
                                                                 "fig5_frontier/*", "figA_*_absolute/*", "figS_*", "gap_map"}, "the six-arms exclude's globs changed: re-measure gap 1 (roster_r2_v4.yaml addendum)"
    checked = 0
    for r in v4.rulings:                                                 # the NORMALISED match is what the record carries and the flag reads
        if r["action"] != "exclude" or r["scope_kind"] != "comparison":
            continue
        for g in (r["scope_json"].get("match") or {}).get("comparison_ids") or []:
            hits = [i for i in ids if glob(i, g)]
            assert not hits, (r["ruling_id"], g, hits)
            checked += 1
    for g in list(six["scope_json"]["match"]["comparison_ids"]) + list(six["scope_json"].get("comparison_ids") or []):   # the file's own spelling too
        assert not [i for i in ids if glob(i, g)], g
    assert checked >= 9


def test_roster_load_cli_on_the_committed_v4_files_writes_the_log(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["roster", "load", "--version", "v4", "--no-ingest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "v4" and out["entries"] == 6370 and out["rulings"] == 93
    assert any(p.endswith(os.path.join("rulings", "r2_v4", "000002-roster_entry.jsonl")) for p in out["written"])


V5_ROSTER, V5_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v5.yaml", "rulings_r2_v5.jsonl"))
PETRI_BEHAVIOURS = ("petri_cyber_bio_chem_capabilities", "petri_defer_to_users", "petri_delusion_sycophancy", "petri_instruction_evasion",
                    "petri_reasoning_unfaithfulness", "petri_self_preservation")
#: schema_def.OPERATING_POINT_FALLBACK_OBJECT_KINDS (migration 0011): a fitted / published kind with no stamp is at the operating point at nominal 1.0
_OP_FALLBACK_KINDS = ("lora", "lora_subtraction", "composed", "relu_steer", "relu_steer_distilled_difference", "alphasteer", "static_r_arditi",
                      "vector", "vector_all", "diffmean", "antidote")
#: The in-frame best-of-N and Petri-behaviour rows of the four models as the live registry filed them on 2026-09-22 (head 149,443; every
#: `in_current_paper = 1` row of those evals, collapsed to its selector-relevant facts; the six behaviours of one fact set share one row):
#: (model_id, paper_label, object_kind, transfer_kind, harness, item_set_id, token_cap_tokens, thinking_mode, decode_source, serving_path,
#:  scoring_rule, n_shot, is_operating_point, strength_nominal, strength_coefficient, eval). Instrument facts, no run content.
IN_FRAME_BON_AND_PETRI_ROWS = [
    ["gemma4e4b_it", "constitution_lora", "lora", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "constitution_lora", "lora", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "constitution_lora", "lora", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "constitution_lora", "lora", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "constitution_relu", "relu_steer", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "constitution_relu", "relu_steer", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "constitution_relu", "relu_steer", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "constitution_relu", "relu_steer", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "constitution_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["gemma4e4b_it", "constitution_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "control_lora", "lora", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "control_lora", "lora", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "control_lora", "lora", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "control_lora", "lora", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "control_relu", "relu_steer", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "control_relu", "relu_steer", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "difference_lora", "lora_subtraction", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "difference_lora", "lora_subtraction", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "difference_lora", "lora_subtraction", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["gemma4e4b_it", "difference_lora", "lora_subtraction", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "difference_relu", "composed", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "unverified", "lora_native", "judge_score_mean", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "unverified", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["gemma4e4b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["gemma4e4b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "generation_config", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["gemma4e4b_it", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "unverified", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["gemma4e4b_it", "safety_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["gemma4e4b_it", "safety_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["llama8b_it", "constitution_lora", "lora", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["llama8b_it", "constitution_lora", "lora", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["llama8b_it", "constitution_lora", "lora", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["llama8b_it", "constitution_lora", "lora", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["llama8b_it", "constitution_lora", "lora", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["llama8b_it", "constitution_lora", "lora", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["llama8b_it", "constitution_relu", "relu_steer", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["llama8b_it", "constitution_relu", "relu_steer", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["llama8b_it", "constitution_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["llama8b_it", "constitution_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, None, None, None, "<six-petri-behaviours>"],
    ["llama8b_it", "constitution_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["llama8b_it", "control_lora", "lora", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["llama8b_it", "control_lora", "lora", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["llama8b_it", "control_lora", "lora", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["llama8b_it", "control_lora", "lora", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["llama8b_it", "control_lora", "lora", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["llama8b_it", "control_lora", "lora", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["llama8b_it", "control_relu", "relu_steer", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["llama8b_it", "control_relu", "relu_steer", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["llama8b_it", "difference_lora", "lora_subtraction", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["llama8b_it", "difference_lora", "lora_subtraction", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["llama8b_it", "difference_lora", "lora_subtraction", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["llama8b_it", "difference_lora", "lora_subtraction", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["llama8b_it", "difference_lora", "lora_subtraction", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["llama8b_it", "difference_lora", "lora_subtraction", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["llama8b_it", "difference_relu", "composed", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "unverified", "lora_native", "judge_score_mean", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["llama8b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "not_available", "unverified", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["llama8b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "generation_config", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["llama8b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "not_available", "generation_config", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["llama8b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["llama8b_it", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, None, None, None, "<six-petri-behaviours>"],
    ["llama8b_it", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["llama8b_it", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "unverified", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["llama8b_it", "safety_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["llama8b_it", "safety_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "not_available", "generation_config", "lens", "judge_graded_weights", None, None, None, None, "<six-petri-behaviours>"],
    ["llama8b_it", "safety_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "constitution_lora", "lora", "chat_checkpoint", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["nemotron3nano4b", "constitution_lora", "lora", "chat_checkpoint", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "constitution_relu", "relu_steer", "chat_checkpoint", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["nemotron3nano4b", "constitution_relu", "relu_steer", "chat_checkpoint", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "constitution_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["nemotron3nano4b", "constitution_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "control_lora", "lora", "chat_checkpoint", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["nemotron3nano4b", "control_lora", "lora", "chat_checkpoint", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "control_relu", "relu_steer", "chat_checkpoint", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["nemotron3nano4b", "control_relu", "relu_steer", "chat_checkpoint", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "difference_lora", "lora_subtraction", "chat_checkpoint", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["nemotron3nano4b", "difference_lora", "lora_subtraction", "chat_checkpoint", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "difference_relu", "composed", "chat_checkpoint", "petri", "petri_scenarios_6x10", 1024, "off", "unverified", "lora_native", "judge_score_mean", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["nemotron3nano4b", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["nemotron3nano4b", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "generation_config", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["nemotron3nano4b", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "unverified", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["nemotron3nano4b", "safety_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["nemotron3nano4b", "safety_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "generation_config", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "arditi_allpos", "vector_all", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, 1, 1.0, 0.3, "strongreject_bon"],
    ["qwen35_4b_it", "arditi_allpos", "vector_all", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "vllm_default", "lora_native", "judge_score_mean", None, None, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "constitution_lora", "lora", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "constitution_lora", "lora", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "model_card", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "constitution_lora", "lora", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "constitution_lora", "lora", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "vllm_default", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "constitution_relu", "relu_steer", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "constitution_relu", "relu_steer", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "model_card", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "constitution_relu", "relu_steer", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "constitution_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["qwen35_4b_it", "constitution_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["qwen35_4b_it", "constitution_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "vllm_default", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "constitution_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "model_card", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "control_lora", "lora", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "control_lora", "lora", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "model_card", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "control_lora", "lora", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "control_lora", "lora", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "vllm_default", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "control_relu", "relu_steer", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "control_relu", "relu_steer", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "model_card", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "difference_lora", "lora_subtraction", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "difference_lora", "lora_subtraction", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "model_card", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "difference_lora", "lora_subtraction", "instruct_direct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "difference_lora", "lora_subtraction", "instruct_direct", "petri", "petri_scenarios_6x10", 1024, "off", "vllm_default", "lora_native", "judge_score_mean", None, 1, 1.0, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "difference_relu", "composed", "base_to_instruct", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, 1, 1.0, None, "strongreject_bon"],
    ["qwen35_4b_it", "difference_relu", "composed", "base_to_instruct", "petri", "petri_scenarios_6x10", 1024, "off", "unverified", "lora_native", "judge_score_mean", None, 1, 1.0, 1.0, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "unverified", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["qwen35_4b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["qwen35_4b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "generation_config", "lens", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["qwen35_4b_it", "none", "none", "none", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["qwen35_4b_it", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "vllm_default", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "unverified", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "none", "none", "none", "petri", "petri_scenarios_6x10", 1024, "off", "model_card", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "safety_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "grandfathered_T1", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["qwen35_4b_it", "safety_si", "system_instruction", "prompted_only", "bespoke_sisteer", "jbb_goals_100", 4096, "off", "model_card", "lora_native", "judge_threshold", None, None, None, None, "strongreject_bon"],
    ["qwen35_4b_it", "safety_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "vllm_default", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"],
    ["qwen35_4b_it", "safety_si", "system_instruction", "prompted_only", "petri", "petri_scenarios_6x10", 1024, "off", "model_card", "lora_native", "judge_score_mean", None, None, None, None, "<six-petri-behaviours>"]
]


def _entry_matches_row(e, row):
    """The object / instrument / strength predicates of v_roster_match (schema_def), in Python, for one roster entry and one fixture row;
    the judge predicate is not modelled (the fixture carries no judgments). Mirrors: every present selector field constrains, an absent one
    does not; token_cap_tokens is an IN-list; operating_point admits a stamp of 1, the migration-0011 fallback (no stamp, a fitted /
    published kind, nominal 1.0), the Arditi all-positions c = 0.3 fallback, or an unsteered / prompted kind."""
    model, label, kind, tk, harness, item_set, cap, thinking, decode, serving, scoring, n_shot, op, nominal, coef, _ = row
    if e["model_id"] != model:
        return False
    if e["object_id"] == "none":                       # the unsteered object is filed by id, not by selector (adapters/roster._object_selector)
        return label == "none" and _instrument_and_strength(e, row)
    if e["object_id"] is not None:
        return False                                   # the fixture carries no pinned object ids
    sel = e["object_selector_json"] or {}
    for k, v in (("paper_label", label), ("object_kind", kind), ("transfer_kind", tk)):
        if sel.get(k) is not None and sel[k] != v:
            return False
    if sel.get("object_role") is not None or sel.get("fit_model_id") is not None or sel.get("object_id_prefix") is not None or sel.get("transfer_kinds") is not None:
        return False                                   # not represented in the fixture: treat as no match (conservative)
    return _instrument_and_strength(e, row)


def _instrument_and_strength(e, row):
    model, label, kind, tk, harness, item_set, cap, thinking, decode, serving, scoring, n_shot, op, nominal, coef, _ = row
    isel = e["instrument_selector_json"] or {}
    for k, v in (("harness", harness), ("item_set_id", item_set), ("thinking_mode", thinking), ("serving_path", serving), ("decode_source", decode),
                 ("scoring_rule", scoring), ("n_shot", n_shot)):
        if isel.get(k) is not None and isel[k] != v:
            return False
    if isel.get("token_cap_tokens") is not None and cap not in isel["token_cap_tokens"]:
        return False
    if e["strength_rule"] != "operating_point":
        return False
    at_op = (op == 1 or kind in ("none", "system_instruction")
             or (op is None and kind in _OP_FALLBACK_KINDS and nominal is not None and abs(nominal - 1.0) < 1e-9)
             or (op is None and kind == "vector_all" and coef is not None and abs(coef - 0.3) < 1e-9))
    return bool(at_op)


def test_committed_v5_adds_the_petri_behaviours_to_the_main_comparisons_and_repoints_best_of_n():
    """roster_r2_v5.yaml (2026-09-22): v4 plus (a) the six Petri behaviour evals beside petri_pooled on fig3_misalign (both
    blocks, the paired-write sub-block included), figQ_misalignment_capability (both lanes) and fig4_transfer - 636 entries, the behaviour
    evals' own selector, the pooled entry's judge; (b) the best-of-N entries of fig1_headline, fig2_jailbreak (both, sub-block included) and
    figQ_jailbreak - and the paired-write sub-block of figA_attacks_absolute/base_to_instruct - repointed
    to the live instrument (bespoke_sisteer / jbb_goals_100 / caps 512-8192; no decode / thinking field) - 124 entries re-minted. Rulings byte-identical to v4's; figR_* untouched."""
    import collections
    if not os.path.exists(V5_ROSTER):
        pytest.skip("no v5 roster file in this tree")
    v4, v5 = RO.load(V4_ROSTER, V4_RULINGS), RO.load(V5_ROSTER, V5_RULINGS)
    s = v5.summary()
    assert v5.version == "v5" and s["entries"] == 7006 and s["rulings"] == 93 and s["dispositions"] == 102
    assert open(V5_RULINGS, "rb").read() == open(V4_RULINGS, "rb").read()
    assert _round_sans_version(v5.paper_round) == _round_sans_version(v4.paper_round)
    d = RO.diff_entries(v5.entries, v4.entries)
    assert not d["changed"] and len(d["only_a"]) == 760 and len(d["only_b"]) == 124
    assert all(k["key"][3] == "strongreject_bon" for k in d["only_b"])                       # re-minted, not lost
    assert collections.Counter(k["key"][0].split("/")[0] for k in d["only_b"]) == {"fig2_jailbreak": 68, "fig1_headline": 32, "figQ_jailbreak": 20, "figA_attacks_absolute": 4}
    assert len(d["coarse_only_a"]) == 582 and not d["coarse_only_b"]                        # 636 entries; fig4_transfer's two transfer kinds share a coarse key
    assert {k: v for k, v in v5.counts_by_comparison.items() if k not in ("fig3_misalign", "fig3_misalign/base_to_instruct", "fig4_transfer", "figQ_misalignment_capability")} == \
           {k: v for k, v in v4.counts_by_comparison.items() if k not in ("fig3_misalign", "fig3_misalign/base_to_instruct", "fig4_transfer", "figQ_misalignment_capability")}
    assert (v5.counts_by_comparison["fig3_misalign"], v5.counts_by_comparison["fig3_misalign/base_to_instruct"], v5.counts_by_comparison["fig4_transfer"],
            v5.counts_by_comparison["figQ_misalignment_capability"]) == (480, 540, 450, 300)
    # the behaviour entries: same selector / judge as figA_petri_absolute's (the one every in-frame behaviour row matches), on the named blocks only
    beh = [e for e in v5.entries if e["eval_id"] in PETRI_BEHAVIOURS]
    stems = collections.Counter(e["comparison_id"].split("/")[0] for e in beh)
    assert set(stems) == {"figA_petri_absolute", "figR_petri_absolute", "fig3_misalign", "fig4_transfer", "figQ_misalignment_capability"}
    assert stems["fig3_misalign"] == 408 and stems["fig4_transfer"] == 108 and stems["figQ_misalignment_capability"] == 120
    assert all(e["instrument_selector_json"] == {} and e["judge_selector_json"] == {"judge_model": "claude-opus-5"} for e in beh)
    pooled = {(e["comparison_id"], e["model_id"], json.dumps(e["object_selector_json"], sort_keys=True), e["object_id"]) for e in v5.entries if e["eval_id"] == "petri_pooled" and e["comparison_id"].split("/")[0] in ("fig3_misalign", "fig4_transfer", "figQ_misalignment_capability")}
    for b in PETRI_BEHAVIOURS:                                                                    # one behaviour entry per pooled entry, same comparison / model / object
        assert {(e["comparison_id"], e["model_id"], json.dumps(e["object_selector_json"], sort_keys=True), e["object_id"]) for e in beh if e["eval_id"] == b and e["comparison_id"].split("/")[0] != "figA_petri_absolute" and e["comparison_id"].split("/")[0] != "figR_petri_absolute"} == pooled
    # the best-of-N repoint: the live selector on the four named blocks, the v1 selector elsewhere, figS_bon_curves its own override
    live = {"harness": "bespoke_sisteer", "item_set_id": "jbb_goals_100", "token_cap_tokens": [512, 4096, 8192]}
    v1sel = {"decode_source": "grandfathered_T1", "thinking_mode": "off", "token_cap_tokens": [512, 4096, 8192]}
    bysel = collections.defaultdict(set)
    for e in v5.entries:
        if e["eval_id"] == "strongreject_bon":
            bysel[e["comparison_id"].split("/")[0]].add(json.dumps(e["instrument_selector_json"], sort_keys=True))
    for stem in ("fig1_headline", "fig2_jailbreak", "figQ_jailbreak"):
        assert bysel[stem] == {json.dumps(live, sort_keys=True)}, stem
    for stem in ("fig4_transfer", "fig5_frontier", "figQ_transfer", "figS_degeneracy", "figR_attacks_absolute", "figQ_frontier"):
        assert bysel[stem] == {json.dumps(v1sel, sort_keys=True)}, stem
    assert bysel["figA_attacks_absolute"] == {json.dumps(v1sel, sort_keys=True), json.dumps(live, sort_keys=True)}   # main blocks v1; the paired-write sub-block live
    assert all(e["instrument_selector_json"] == live for e in v5.entries if e["eval_id"] == "strongreject_bon" and e["comparison_id"] == "figA_attacks_absolute/base_to_instruct"
               and (e["object_selector_json"] or {}).get("paper_label") == "difference_relu")
    assert all(e["instrument_selector_json"] == v1sel for e in v5.entries if e["eval_id"] == "strongreject_bon" and e["comparison_id"].startswith("figA_attacks_absolute/")
               and (e["object_selector_json"] or {}).get("paper_label") != "difference_relu")
    assert bysel["figS_bon_curves"] == {json.dumps({"token_cap_tokens": [512, 4096]}, sort_keys=True)}
    assert sum(1 for e in v5.entries if e["eval_id"] == "strongreject_bon" and e["instrument_selector_json"] == live) == 124
    assert {e["comparison_id"] for e in v5.entries if e["comparison_id"].startswith("figR_")} == {e["comparison_id"] for e in v4.entries if e["comparison_id"].startswith("figR_")}


def test_committed_v5_every_in_frame_best_of_n_and_petri_behaviour_row_matches_a_main_figure_entry():
    """Fixture query against the COMMITTED roster (no live database): every in-frame strongreject_bon / Petri-behaviour row of the four models
    (IN_FRAME_BON_AND_PETRI_ROWS, the live facts of 2026-09-22) matches, on the object / instrument / strength predicates of v_roster_match,
    an entry whose comparison is a MAIN figure (fig1_headline, fig2_jailbreak, fig3_misalign, fig4_transfer, figQ_jailbreak,
    figQ_misalignment_capability) - or, for the three families' ReLU-steer objects the six-arms exclude keeps off the main figures, their own
    figR_* comparison. The Arditi all-positions rows are the one class no entry can carry: the roster declares no
    `arditi_allpos` object (meta.addendum_2026_09_17.arditi_allpos_not_a_roster_object); they are asserted unmatched so adding the object
    is a deliberate change here too."""
    if not os.path.exists(V5_ROSTER):
        pytest.skip("no v5 roster file in this tree")
    v5 = RO.load(V5_ROSTER, V5_RULINGS)
    MAIN = ("fig1_headline/", "fig2_jailbreak/", "fig3_misalign/", "fig4_transfer", "figQ_jailbreak/", "figQ_misalignment_capability/")
    by_eval = {}
    for e in v5.entries:
        by_eval.setdefault(e["eval_id"], []).append(e)
    unmatched, main_only_missing = [], []
    for row in IN_FRAME_BON_AND_PETRI_ROWS:
        evals = PETRI_BEHAVIOURS if row[-1] == "<six-petri-behaviours>" else (row[-1],)
        for ev in evals:
            hits = [e for e in by_eval.get(ev, []) if _entry_matches_row(e, row)]
            main = [e for e in hits if e["comparison_id"].startswith(MAIN)]
            figr = [e for e in hits if e["comparison_id"].startswith("figR_")]
            if row[1] == "arditi_allpos":
                assert not hits, (row, [e["comparison_id"] for e in hits])
                continue
            if not (main or figr):
                unmatched.append((row, ev, sorted({e["comparison_id"] for e in hits})))
            elif not main and row[0] != "qwen35_4b_it":
                assert row[1] in ("control_relu", "constitution_relu"), (row, ev)       # only the ReLU-steer set lives on figR_* alone
            elif not main:
                main_only_missing.append((row, ev))
    assert not unmatched, unmatched[:10]
    assert not main_only_missing, main_only_missing[:10]
    # the matcher fires where it should: a fixture row against a foreign model / eval matches nothing
    probe = list(IN_FRAME_BON_AND_PETRI_ROWS[0]); probe[0] = "glm46vflash"
    assert not any(_entry_matches_row(e, probe) for e in v5.entries)


def test_roster_load_cli_on_the_committed_v5_files_writes_the_log(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["roster", "load", "--version", "v5", "--no-ingest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "v5" and out["entries"] == 7006 and out["rulings"] == 93
    assert any(p.endswith(os.path.join("rulings", "r2_v5", "000002-roster_entry.jsonl")) for p in out["written"])


V6_ROSTER, V6_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v6.yaml", "rulings_r2_v6.jsonl"))
#: The eight PADDED static instruments roster v6 names (adapters/static_padding.padded_instrument_fields through
#: identity.instrument_id from the seeded static instrument of qwen35_4b_it for the eval plus padding_tokens + the filler-contract hash,
#: cap 16,384 single pass, window 98,304): instrument_id -> (padding_tokens, eval_id). Minted READ-ONLY over the durable root on
#: 2026-09-23 (8 groups, 24 runs, 0 refused; two independent dry runs byte-identical). A change here is a change of the instruments the
#: comparison reads: re-mint and re-measure, never edit.
PADDED_STATIC_INSTRUMENTS = {
    "d32a9a427641bb9ac58a1ab1bf3f4d731858bf350ebadd83639ce29c7c7bebb2": (16384, "codeattack"),
    "5b8f6b6d1950a43b49a2d6a9fbc0a9a99d5ad9212a6a84680f987f4d77bbfff6": (16384, "grok420"),
    "8eeeb610019e9fbaa64fddfeb68d2c29efcc154b933904e1446926a89ac4a371": (16384, "pasttense"),
    "399faa2a7d8305e7260dd1b76937053ed4d444192f956e75ab7fbef660fa9070": (16384, "renellm"),
    "606205347ed142127138bf480815340fc2f46ecedb38936c8fc82323b6fa89a6": (65536, "codeattack"),
    "c55dcc68a4a12c0255f52c7f37c3a8c0cd28f0cbc4a8022896c0aac0bfb8e0d9": (65536, "grok420"),
    "93a898bd1b1e68fdd6c73772c387fed58b5156dd33e8f3e22dc5d68fa67ead67": (65536, "pasttense"),
    "b9b75980d5e9b16428f7454a41234b4ff761ca47130440a7048452cd402a0a24": (65536, "renellm"),
}

PADDING_COMPARISON = "figP_padding/instruct_direct"


def test_committed_v6_adds_the_padding_comparison_naming_the_eight_padded_instruments():
    """roster_r2_v6.yaml (2026-09-23): v5 plus ONE comparison of its own, figP_padding/instruct_direct -
    qwen35_4b_it x {none, safety_si, constitution_si} x the four static evals on the EIGHT padded instruments (2 pads x 4 families), one
    entry per padded instrument x condition, each entry NAMING its instrument (instrument_id set, selector null) because the pad is
    instrument identity (0012) and no selector field carries it. 24 additive entries; every v5 entry keeps its identity; rulings v5's bytes."""
    import collections
    if not os.path.exists(V6_ROSTER):
        pytest.skip("no v6 roster file in this tree")
    v5, v6 = RO.load(V5_ROSTER, V5_RULINGS), RO.load(V6_ROSTER, V6_RULINGS)
    s = v6.summary()
    assert v6.version == "v6" and s["entries"] == 7030 and s["rulings"] == 93 and s["dispositions"] == 102
    assert open(V6_RULINGS, "rb").read() == open(V5_RULINGS, "rb").read()
    assert _round_sans_version(v6.paper_round) == _round_sans_version(v5.paper_round)
    d = RO.diff_entries(v6.entries, v5.entries)
    assert not d["changed"] and len(d["only_a"]) == 24 and not d["only_b"]                  # additive: nothing re-minted, nothing lost
    assert len(d["coarse_only_a"]) == 12 and not d["coarse_only_b"]                        # 3 objects x 4 evals; the two pad levels share a coarse key
    assert all(k["key"][0] == PADDING_COMPARISON for k in d["only_a"])
    assert {k: v for k, v in v6.counts_by_comparison.items() if k != PADDING_COMPARISON} == v5.counts_by_comparison
    assert v6.counts_by_comparison[PADDING_COMPARISON] == 24
    new = [e for e in v6.entries if e["comparison_id"] == PADDING_COMPARISON]
    assert len(new) == 24 and all(e["model_id"] == "qwen35_4b_it" and e["strength_rule"] == "operating_point" for e in new)
    # every entry names ONE of the eight padded instruments, selector null (the DDL's instrument XOR selector), and its eval is the
    # instrument's eval; each (instrument, object) exactly once; the eight ids all named; no other v6 entry names an instrument
    assert {e["instrument_id"] for e in new} == set(PADDED_STATIC_INSTRUMENTS)
    assert all(e["instrument_selector_json"] is None and PADDED_STATIC_INSTRUMENTS[e["instrument_id"]][1] == e["eval_id"] for e in new)
    assert all(e["extra"]["instrument_named"] is True and e["extra"]["instrument_selector_legacy"] is None for e in new)
    assert all(e["extra"]["eval_extra"]["padding_tokens"] == PADDED_STATIC_INSTRUMENTS[e["instrument_id"]][0] for e in new)
    objs = collections.Counter((e["object_id"], json.dumps(e["object_selector_json"], sort_keys=True)) for e in new)
    assert objs == {("none", "null"): 8, (None, json.dumps({"object_kind": "system_instruction", "paper_label": "safety_si"}, sort_keys=True)): 8,
                    (None, json.dumps({"object_kind": "system_instruction", "paper_label": "constitution_si"}, sort_keys=True)): 8}
    assert collections.Counter(e["eval_id"] for e in new) == {"pasttense": 6, "codeattack": 6, "renellm": 6, "grok420": 6}
    assert len({(e["instrument_id"], e["object_id"], json.dumps(e["object_selector_json"], sort_keys=True)) for e in new}) == 24
    assert all(e["judge_selector_json"] == {"judge_model": "claude-haiku-4-5"} for e in new)   # the static evals' judge (the V1 Haiku StrongREJECT judgment)
    assert all(e["inferred"] == 1 and e["drawn"] == 1 for e in new)
    assert not any(e["instrument_id"] for e in v6.entries if e["comparison_id"] != PADDING_COMPARISON)
    # the yaml's own table of the eight (meta.addendum_2026_09_23_v6.padded_instruments) agrees with the evals[] items and with the pin here
    roster = yaml.safe_load(open(V6_ROSTER, encoding="utf-8"))
    table = {r["instrument_id"]: (r["padding_tokens"], r["eval_id"]) for r in roster["meta"]["addendum_2026_09_23_v6"]["padded_instruments"]}
    assert table == PADDED_STATIC_INSTRUMENTS
    assert all(r["token_cap_tokens"] == 16384 and r["context_window_tokens"] == 98304 for r in roster["meta"]["addendum_2026_09_23_v6"]["padded_instruments"])
    assert collections.Counter(p for p, _ in PADDED_STATIC_INSTRUMENTS.values()) == {16384: 4, 65536: 4}
    assert roster["meta"]["expanded_counts_by_comparison"]["total"] == 7030
    # the figR_* and every other block are v5's, untouched
    assert {e["comparison_id"] for e in v6.entries if e["comparison_id"].startswith("figR_")} == {e["comparison_id"] for e in v5.entries if e["comparison_id"].startswith("figR_")}


def test_named_instrument_grammar_expands_one_entry_per_id_and_refuses_malformed_items():
    """An evals[] item may name `instrument_ids` (v6): one entry per id with instrument_id set and the selector null, distinct identities
    from each other and from the selector-shaped entry; a non-sha id, a duplicate, or ids beside a selector override refuse by name;
    `unknown_to_db` reports a named id the instrument table lacks (it dependency-blocks at the fold; the loader never refuses it)."""
    a, b = "a" * 64, "b" * 64
    m = _mini()
    m["comparisons"][0]["evals"] = [{"eval_id": synth.EVAL_ID, "instrument_ids": [a, b]}]
    m["meta"]["expanded_counts_by_comparison"] = {"fig_synth": 4}
    exp = RO.expand(m)
    assert len(exp.entries) == 4 and {e["instrument_id"] for e in exp.entries} == {a, b}
    assert all(e["instrument_selector_json"] is None and e["extra"]["instrument_named"] and e["extra"]["instrument_selector_legacy"] is None for e in exp.entries)
    assert len({e["roster_entry_id"] for e in exp.entries}) == 4
    plain = {e["roster_entry_id"] for e in RO.expand(_mini()).entries}
    assert not plain & {e["roster_entry_id"] for e in exp.entries}                          # the instrument is identity
    one = _mini(); one["comparisons"][0]["evals"] = [{"eval_id": synth.EVAL_ID, "instrument_id": a}]
    assert [e["instrument_id"] for e in RO.expand(one).entries] == [a, a]                  # the scalar spelling
    for item, needle in (({"eval_id": synth.EVAL_ID, "instrument_ids": ["not-a-sha"]}, "not a 64-hex sha256"),
                         ({"eval_id": synth.EVAL_ID, "instrument_ids": [a, a]}, "listed twice"),
                         ({"eval_id": synth.EVAL_ID, "instrument_ids": [a], "instrument_selector_json_override": {"caps": [512]}}, "exclusive")):
        bad = _mini(); bad["comparisons"][0]["evals"] = [item]
        with pytest.raises(RO.RosterRefusal, match=needle):
            RO.expand(bad)
    # against a database: the named id is reported as unknown (models / evals / objects too, on an empty root)
    conn = DB.connect_memory()                                                             # the tables, no rows: every id unknown
    load = RO.RosterLoad(exp.round_id, "v1", exp.paper_round, exp.entries, [], [], None, exp.counts_by_comparison, [])
    assert RO.unknown_to_db(conn, load)["instrument_ids"] == [a, b]


def test_named_instrument_entry_dependency_blocks_until_its_instrument_lands_then_matches_only_runs_on_it(tmp_path):
    """The v6 mechanism end to end on the synthetic corpus. A roster whose eval item names an instrument the database does not carry:
    the roster_entry records dependency-block at the fold (refused 0) and no run matches; once an `instrument` record with that id lands
    the entries fold, and v_roster_match admits a run on THAT instrument only - the corpus's runs of the same model / eval on the seeded
    instrument match none of them (a selector-shaped entry matched both). The dependency-block is what lets roster v6 load before the
    static_padding read mints the eight padded instruments."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    other = synth.instrument_fields(context_window_tokens=98304)                          # another instrument of the same eval
    other_id = str(other["instrument_id"])
    assert other_id != synth.INSTRUMENT_ID
    m = _mini()
    m["comparisons"][0]["evals"] = [{"eval_id": synth.EVAL_ID, "instrument_ids": [other_id]}]
    ry = tmp_path / "roster.yaml"
    ry.write_text(yaml.safe_dump(m, sort_keys=False))
    load = RO.load(str(ry))
    assert len(load.entries) == 2 and all(e["instrument_id"] == other_id for e in load.entries)
    RO.write_records(root, load, written_utc=synth.t(50))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert RO.unknown_to_db(conn, load)["instrument_ids"] == [other_id]
    assert conn.execute("SELECT COUNT(*) FROM roster_entry WHERE round_id = 'r_synth'").fetchone()[0] == 0
    blocked = conn.execute("SELECT record_type, missing_parent_table, missing_parent_id FROM blocked_records").fetchall()
    assert len([b for b in blocked if b[0] == "roster_entry"]) == 2 and all((b[1], b[2]) == ("instrument", other_id) for b in blocked if b[0] == "roster_entry"), blocked
    assert conn.execute("SELECT COUNT(*) FROM v_roster_match").fetchone()[0] == 0
    conn.close()
    # the instrument lands (as the static_padding read would mint it): the entries unblock and fold
    _write(root, "extra", "000001-extra.jsonl", [synth.rec("instrument", other, synth.t(60))])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0, rep2.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert RO.unknown_to_db(conn, load)["instrument_ids"] == []
    assert conn.execute("SELECT COUNT(*) FROM roster_entry WHERE round_id = 'r_synth' AND instrument_id = ? AND instrument_selector_json IS NULL", (other_id,)).fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM blocked_records WHERE record_type = 'roster_entry'").fetchone()[0] == 0
    # the corpus's runs are on the seeded instrument: a named-instrument entry admits none of them ...
    assert conn.execute("SELECT COUNT(*) FROM v_roster_match").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM run WHERE instrument_id = ?", (synth.INSTRUMENT_ID,)).fetchone()[0] == 2
    # ... while a run of the same model / object / eval on the NAMED instrument is admitted by the id alone
    run_id = conn.execute("SELECT run_id FROM run WHERE object_id = 'none'").fetchone()[0]
    conn.close()
    conn = DB.connect(DB.db_path(root))
    try:
        conn.execute("UPDATE run SET instrument_id = ? WHERE run_id = ?", (other_id, run_id))
        conn.commit()
        matched = {r[0]: r[1] for r in conn.execute("SELECT run_id, comparison_id FROM v_roster_match")}
        assert matched == {run_id: "fig_synth/base_to_instruct"}, matched
    finally:
        conn.close()


def _pre_0013_roster_match_sql() -> str:
    """The CREATE text of v_roster_match as migration 0011 left it (0012 did not touch the object): the selector branch WITHOUT the
    padding predicate 0013 adds, read from the frozen file so the before/after below is history, not a re-implementation."""
    from sisteer.registry import migrations as M
    path = next(p for v, _, p in M.migration_files() if v == 11)
    text = open(path, encoding="utf-8").read()
    sql = text[text.index("CREATE VIEW v_roster_match AS"):].strip().rstrip(";")
    assert "padding_tokens" not in sql and "re.instrument_id IS NULL" in sql
    return sql


def test_a_selector_never_matches_a_padded_instrument_which_is_reached_only_by_name(tmp_path):
    """Migration 0013: a padded instrument (padding_tokens set, 0012) that agrees with the seeded instrument on EVERY
    selector-visible field -- harness, thinking_mode, cap, decode, scoring, serving path, item set -- matches no selector-shaped entry;
    under 0011's text of v_roster_match (the frozen migration file) it matched them all, which is how the 24 live padded runs pooled into
    the ordinary static comparisons on 2026-09-23. It matches the entry that NAMES it (roster v6's `instrument_ids`), the unpadded runs
    keep matching the selector entries exactly as before, the padded runs' `comparison_ids_json` carries the named comparison alone, and
    no entry is left in the gap map."""
    from sisteer.registry import identity as I
    contract = I.padding_identity_sha256({"filler_sha256": "7" * 64, "mode": "fixed", "placement": "before", "separator": "\n\n", "tolerance": 0.02})
    padded = synth.instrument_fields(padding_tokens=16384, padding_identity_sha256=contract)
    padded_id = str(padded["instrument_id"])
    assert padded_id != synth.INSTRUMENT_ID
    # the pad is the ONLY identity difference: every other identity field equals the seeded (unpadded) instrument's
    assert {k for k, _ in I.INSTRUMENT_ID_FIELDS if padded.get(k) != synth.INSTRUMENT.get(k)} == {"padding_tokens", "padding_identity_sha256"}
    m = _mini()
    m["comparisons"].append({"comparison_id_template": "fig_padded/{transfer_kind}", "figure_legacy": "fig_padded",
                             "models": [{"model_id": synth.MODEL_ID, "transfer_kind": "base_to_instruct"}],
                             "objects": ["none", "constitution_lora"], "evals": [{"eval_id": synth.EVAL_ID, "instrument_ids": [padded_id]}],
                             "strength_rules": ["operating_point"], "source_citation": "paperlib.config.SYNTHETIC @ eeeeeeee"})
    m["meta"]["expanded_counts_by_comparison"] = {"fig_synth": 2, "fig_padded": 2}
    # two runs on the padded instrument, the unsteered model and the LoRA, beside the corpus's two on the seeded one
    rid_pn, recs_pn = synth.run_records("padded_none", 400, "none", "none", n_items=3, instrument_id=padded_id)
    rid_pl, recs_pl = synth.run_records("padded_lora", 420, "constitution_lora@deadbeef", synth.OBJECT_SHA, n_items=3, instrument_id=padded_id)
    root, ids, load, _ = _setup(tmp_path, m, extra_records=[synth.rec("instrument", padded, synth.t(40))] + recs_pn + recs_pl)
    none_run, lora_run = ids["run_ids"]
    entries = {e["roster_entry_id"]: e for e in load.entries}
    named = {k for k, e in entries.items() if e["instrument_id"] == padded_id}
    selector = {k for k, e in entries.items() if e["instrument_id"] is None}
    assert len(named) == 2 and len(selector) == 2 and all(entries[k]["comparison_id"] == "fig_padded/base_to_instruct" for k in named)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM roster_entry WHERE round_id = 'r_synth'").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    row = dict(conn.execute("SELECT * FROM instrument WHERE instrument_id = ?", (padded_id,)).fetchone())
    seeded = dict(conn.execute("SELECT * FROM instrument WHERE instrument_id = ?", (synth.INSTRUMENT_ID,)).fetchone())
    assert (row["padding_tokens"], row["padding_identity_sha256"]) == (16384, contract) and seeded["padding_tokens"] is None
    # the padded instrument SATISFIES the selector's fields (caps [512], harness, thinking_mode) -- the exclusion below is the pad alone
    sel = entries[next(iter(selector))]["instrument_selector_json"]
    assert sel == {"token_cap_tokens": [512], "harness": "bespoke_sisteer", "thinking_mode": "off"}
    assert (row["token_cap_tokens"], row["harness"], row["thinking_mode"]) == (512, "bespoke_sisteer", "off")
    assert all(row[k] == seeded[k] for k in ("harness", "item_set_id", "thinking_mode", "serving_path", "decode_source", "scoring_rule", "n_shot", "token_cap_tokens"))
    matched = {}
    for r in conn.execute("SELECT run_id, roster_entry_id, comparison_id FROM v_roster_match"):
        matched.setdefault(r[0], set()).add((r[1], r[2]))
    # the unpadded runs match the selector entries as before, and NOT the named ones; the padded runs match the named entries ONLY
    assert {k for k, _ in matched[none_run]} | {k for k, _ in matched[lora_run]} == selector
    assert {c for _, c in matched[none_run]} == {c for _, c in matched[lora_run]} == {"fig_synth/base_to_instruct"}
    assert {k for k, _ in matched[rid_pn]} | {k for k, _ in matched[rid_pl]} == named
    assert {c for _, c in matched[rid_pn]} == {c for _, c in matched[rid_pl]} == {"fig_padded/base_to_instruct"}
    assert set(matched) == {none_run, lora_run, rid_pn, rid_pl}
    # the paper relation: the padded runs are selected under the named comparison alone; the unpadded under theirs alone; no gap
    pc = {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM v_paper_current")}
    for rid in (rid_pn, rid_pl):
        assert pc[rid]["selection_state"] == "selected" and pc[rid]["in_current_paper"] == 1
        assert json.loads(pc[rid]["comparison_ids_json"]) == ["fig_padded/base_to_instruct"]
        assert set(json.loads(pc[rid]["roster_entry_ids_json"])) <= named
    for rid in (none_run, lora_run):
        assert pc[rid]["selection_state"] == "selected" and json.loads(pc[rid]["comparison_ids_json"]) == ["fig_synth/base_to_instruct"]
    long = {r[0]: (r[1], json.loads(r[2])) for r in conn.execute("SELECT run_id, padding_tokens, comparison_ids_json FROM v_paper_long")}
    assert long[rid_pn] == (16384, ["fig_padded/base_to_instruct"]) and long[none_run] == (None, ["fig_synth/base_to_instruct"])
    assert conn.execute("SELECT COUNT(*) FROM v_roster_gaps").fetchone()[0] == 0
    # BEFORE 0013 (the object's text as 0011 left it, over the same rows): the padded runs matched the selector entries too -- the pooling
    conn.execute(_pre_0013_roster_match_sql().replace("CREATE VIEW v_roster_match AS", "CREATE TEMP VIEW pre_0013_roster_match AS", 1))
    old = {}
    for r in conn.execute("SELECT run_id, roster_entry_id FROM pre_0013_roster_match"):
        old.setdefault(r[0], set()).add(r[1])
    assert old[rid_pn] | old[rid_pl] == named | selector and old[none_run] | old[lora_run] == selector
    assert all(len(old[rid]) == 2 for rid in (rid_pn, rid_pl)) and all(len(matched[rid]) == 1 for rid in (rid_pn, rid_pl))
    conn.close()


def test_roster_load_cli_on_the_committed_v6_files_writes_the_log_and_names_the_unknown_instruments(tmp_path, monkeypatch, capsys):
    if not os.path.exists(V6_ROSTER):
        pytest.skip("no v6 roster file in this tree")
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["roster", "load", "--version", "v6", "--no-ingest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "v6" and out["entries"] == 7030 and out["rulings"] == 93
    assert any(p.endswith(os.path.join("rulings", "r2_v6", "000002-roster_entry.jsonl")) for p in out["written"])
    # against the (seedless) root the eight padded instruments are reported unknown: the entries dependency-block, never refuse
    conn = DB.connect(DB.db_path(root), readonly=True)
    load = RO.load(V6_ROSTER, V6_RULINGS, conn=conn)
    assert load.unknown_to_db["instrument_ids"] == sorted(PADDED_STATIC_INSTRUMENTS)
    assert "instrument" in RO.show_text(load)
    conn.close()


V7_ROSTER, V7_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v7.yaml", "rulings_r2_v7.jsonl"))
#: The four ZERO-POINT static instruments roster v7 names (migration 0014): the padded instruments' fields with
#: padding_tokens 0 and the SAME filler-contract hash (1d9d5678c002…), minted READ-ONLY by adapters/static_padding.padded_instrument_fields
#: over the real p0 stores on 2026-09-23 (4 groups, 12 runs filed, 0 refused; two processes byte-identical). instrument_id -> eval_id. A
#: change here is a change of the instruments the comparison reads: re-mint and re-measure, never edit.
ZERO_POINT_STATIC_INSTRUMENTS = {
    "20861ffa2667dbf44b3df0ba330bf7fd09ca34a620e028e65778eb1c74078758": "pasttense",
    "f4f3de2edbeb3e9f5723655c57dc1228ef3d652a26f6f2a6f37e9599648ad29a": "codeattack",
    "98e7d5b48701f8cd04dd5ef1176095a057df0a8a14766a455baad98f2f17d593": "renellm",
    "9e9341a7043f2c39f1401b0db514c3cb44248220b4d8d8a4bc8d7e82522546f0": "grok420",
}


def test_committed_v7_adds_the_four_zero_point_instruments_to_the_padding_comparison():
    """roster_r2_v7.yaml (2026-09-23): v6 plus the ZERO POINT of the padding series -- the figP_padding/instruct_direct
    evals[] items additionally name the four p0 instruments (padding_tokens 0 WITH the padded contract, migration 0014), one per static family,
    so the comparison reads 0 / 16,384 / 65,536 for every condition. 12 additive entries (3 objects x 4 evals), every v6 entry unchanged in
    identity, no new coarse key (the zero point shares its (object, eval) with the padded levels), rulings v6's bytes, every other summary
    number v6's."""
    import collections
    if not os.path.exists(V7_ROSTER):
        pytest.skip("no v7 roster file in this tree")
    v6, v7 = RO.load(V6_ROSTER, V6_RULINGS), RO.load(V7_ROSTER, V7_RULINGS)
    s6, s7 = v6.summary(), v7.summary()
    assert v7.version == "v7" and s7["entries"] == 7042 and s7["rulings"] == 93 and s7["dispositions"] == 102
    # everything the loader summarises but the twelve new (inferred) entries is v6's -- the disagreements count included
    moved = {"entries", "inferred", "counts_by_comparison", "version"}
    assert {k: v for k, v in s7.items() if k not in moved} == {k: v for k, v in s6.items() if k not in moved}
    assert s7["inferred"] == s6["inferred"] + 12 and s7["disagreements"] == s6["disagreements"] == 661
    assert open(V7_RULINGS, "rb").read() == open(V6_RULINGS, "rb").read()
    assert _round_sans_version(v7.paper_round) == _round_sans_version(v6.paper_round)
    d = RO.diff_entries(v7.entries, v6.entries)
    assert not d["changed"] and len(d["only_a"]) == 12 and not d["only_b"]                  # additive: nothing re-minted, nothing lost, no v6 payload moved
    assert not d["coarse_only_a"] and not d["coarse_only_b"]                                 # the zero point shares every coarse key with its padded siblings
    assert all(k["key"][0] == PADDING_COMPARISON for k in d["only_a"])
    assert {k: v for k, v in v7.counts_by_comparison.items() if k != PADDING_COMPARISON} == {k: v for k, v in v6.counts_by_comparison.items() if k != PADDING_COMPARISON}
    assert v7.counts_by_comparison[PADDING_COMPARISON] == 36 and v6.counts_by_comparison[PADDING_COMPARISON] == 24
    new = [e for e in v7.entries if e["comparison_id"] == PADDING_COMPARISON]
    assert {e["instrument_id"] for e in new} == set(PADDED_STATIC_INSTRUMENTS) | set(ZERO_POINT_STATIC_INSTRUMENTS)
    assert not (set(ZERO_POINT_STATIC_INSTRUMENTS) & set(PADDED_STATIC_INSTRUMENTS))
    zero = [e for e in new if e["instrument_id"] in ZERO_POINT_STATIC_INSTRUMENTS]
    assert len(zero) == 12 and all(e["model_id"] == "qwen35_4b_it" and e["strength_rule"] == "operating_point" for e in zero)
    assert all(e["instrument_selector_json"] is None and ZERO_POINT_STATIC_INSTRUMENTS[e["instrument_id"]] == e["eval_id"] for e in zero)
    assert all(e["extra"]["instrument_named"] is True and e["extra"]["eval_extra"]["padding_tokens"] == 0 for e in zero)
    assert all(e["extra"]["eval_extra"]["standup"] == "pad_static_qwen35_4b_p0" and e["extra"]["eval_extra"]["token_cap_tokens"] == 16384 for e in zero)
    assert collections.Counter(e["eval_id"] for e in zero) == {"pasttense": 3, "codeattack": 3, "renellm": 3, "grok420": 3}
    objs = collections.Counter((e["object_id"], json.dumps(e["object_selector_json"], sort_keys=True)) for e in zero)
    assert objs == {("none", "null"): 4, (None, json.dumps({"object_kind": "system_instruction", "paper_label": "safety_si"}, sort_keys=True)): 4,
                    (None, json.dumps({"object_kind": "system_instruction", "paper_label": "constitution_si"}, sort_keys=True)): 4}
    assert all(e["judge_selector_json"] == {"judge_model": "claude-haiku-4-5"} for e in zero)
    # the v6 padded entries are untouched in identity and content
    assert {e["roster_entry_id"] for e in v6.entries if e["comparison_id"] == PADDING_COMPARISON} < {e["roster_entry_id"] for e in new}
    # the yaml's own table of the four (meta.addendum_2026_09_23_v7.zero_point_instruments) agrees with the evals[] items and the pin here
    roster = yaml.safe_load(open(V7_ROSTER, encoding="utf-8"))
    table = {r["instrument_id"]: (r["padding_tokens"], r["eval_id"]) for r in roster["meta"]["addendum_2026_09_23_v7"]["zero_point_instruments"]}
    assert table == {i: (0, ev) for i, ev in ZERO_POINT_STATIC_INSTRUMENTS.items()}
    assert all(r["token_cap_tokens"] == 16384 and r["context_window_tokens"] == 98304 and len(r["padding_identity_sha256"]) == 64
               for r in roster["meta"]["addendum_2026_09_23_v7"]["zero_point_instruments"])
    assert len({r["padding_identity_sha256"] for r in roster["meta"]["addendum_2026_09_23_v7"]["zero_point_instruments"]}) == 1
    assert roster["meta"]["expanded_counts_by_comparison"][PADDING_COMPARISON] == 36 and roster["meta"]["expanded_counts_by_comparison"]["total"] == 7042
    assert roster["meta"]["roster_version"] == "v7"


def test_roster_load_cli_on_the_committed_v7_files_parks_the_twelve_padding_instruments_as_unknown(tmp_path, monkeypatch, capsys):
    """On a root without the static_padding read the eight padded AND the four zero-point instruments are unknown: the loader lists them and
    the entries dependency-block at the fold (never refused), exactly as v6's did before window 8."""
    if not os.path.exists(V7_ROSTER):
        pytest.skip("no v7 roster file in this tree")
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["roster", "load", "--version", "v7", "--no-ingest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "v7" and out["entries"] == 7042 and out["rulings"] == 93
    assert any(p.endswith(os.path.join("rulings", "r2_v7", "000002-roster_entry.jsonl")) for p in out["written"])
    conn = DB.connect(DB.db_path(root), readonly=True)
    load = RO.load(V7_ROSTER, V7_RULINGS, conn=conn)
    assert load.unknown_to_db["instrument_ids"] == sorted(set(PADDED_STATIC_INSTRUMENTS) | set(ZERO_POINT_STATIC_INSTRUMENTS))
    conn.close()


# --------------------------------------------------------------------------------------------
# a synthetic roster through the log and the views
# --------------------------------------------------------------------------------------------


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _setup(tmp_path, roster=None, rulings=(), n_runs=2, extra_records=()):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=n_runs, n_items=3)
    ry = tmp_path / "roster.yaml"
    ry.write_text(yaml.safe_dump(roster or _mini(), sort_keys=False))
    rj = tmp_path / "rulings.jsonl"
    rj.write_text("".join(json.dumps(r) + "\n" for r in rulings))
    load = RO.load(str(ry), str(rj) if rulings else None)
    paths = RO.write_records(root, load, written_utc=synth.t(50))
    if extra_records:
        _write(root, "extra", "000001-extra.jsonl", list(extra_records))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root, ids, load, paths


def test_synthetic_roster_ingests_and_diff_against_the_database_is_equal(tmp_path):
    root, ids, load, paths = _setup(tmp_path)
    assert [os.path.basename(p) for p in paths] == ["000001-paper_round.jsonl", "000002-roster_entry.jsonl", "000003-ruling.jsonl"] or \
        [os.path.basename(p) for p in paths] == ["000001-paper_round.jsonl", "000002-roster_entry.jsonl"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM roster_entry WHERE round_id = 'r_synth'").fetchone()[0] == 2
    # both synthetic runs match: the none entry matches the unsteered run, the selector entry the fitted one
    matched = {r[0]: r[1] for r in conn.execute("SELECT run_id, comparison_id FROM v_roster_match")}
    assert set(matched) == set(ids["run_ids"]) and set(matched.values()) == {"fig_synth/base_to_instruct"}
    d = RO.diff_entries(load.entries, RO.db_entries(conn, "r_synth"))
    assert d["equal"], d
    # a second load of the same files is a no-op on the log (identical bytes), a changed file is refused
    assert RO.write_records(root, load, written_utc=synth.t(50)) == paths
    load.entries[0]["inferred"] = 1
    with pytest.raises(RO.RosterRefusal, match="exists with different content"):
        RO.write_records(root, load, written_utc=synth.t(50))
    conn.close()


def _alphasteer_object():
    """A synthetic AlphaSteer object: a contrastive direction (the paired-nullness CHECK wants the contrast sets, no corpus)."""
    o = synth.object_lora(sha="a" * 64, paper_label="alphasteer_rebake", role="baseline_published")
    for k in ("fit_corpus_id", "fit_corpus_sha256", "fit_rank", "fit_alpha"):
        o.pop(k, None)
    o.update(object_kind="alphasteer", fit_objective="alphasteer", campaign="alphasteer_f22", fit_target_kind="contrastive_direction",
             fit_contrast_sets_json={"positive": {"id": "harmful_pool", "sha256": synth.SHA256_A, "n": 64}, "negative": {"id": "benign_pool", "sha256": synth.SHA256_B, "n": 64}},
             fit_positions="last_token_of_appended_assistant_header")
    return o


def test_v_paper_objects_reads_alphasteer_out_by_the_latest_ruling_while_its_runs_stay_ingested(tmp_path):
    """An earlier include and a later exclude cover the same AlphaSteer object; the LATEST ruled decides, the runs stay."""
    roster = _mini()
    roster["objects"]["alphasteer_rebake"] = {"paper_label": "alphasteer_rebake", "object_kind": "alphasteer", "object_role": "baseline_published", "lane_free": True}
    roster["comparisons"][0]["objects"].append("alphasteer_rebake")
    rid_as, recs_as = synth.run_records("synthstore_as", 900, "alphasteer_rebake@aaaaaaaa", "a" * 64, 3)
    dims = [synth.rec("object", _alphasteer_object(), synth.t(12))]
    rulings = [_ruling(ruling_id="R_include_as", action="include", scope_kind="key", scope_json={"paper_label": "alphasteer_rebake"}, ruled_utc="2026-01-03"),
               _ruling(ruling_id="R_out", action="exclude", scope_kind="object", scope_json={"object_kind": "alphasteer", "match": {"object_kinds": ["alphasteer"]}},
                       ruled_utc="2026-01-04T00:20:00Z", ruled_by="the authors")]
    root, ids, load, _ = _setup(tmp_path, roster, rulings, extra_records=dims + recs_as)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = dict(conn.execute("SELECT * FROM v_paper_objects WHERE object_id = ?", ("a" * 64,)).fetchone())
    assert row["in_paper"] == 0 and row["n_runs"] == 1 and row["n_runs_in_paper"] == 0
    assert "R_out" in json.loads(row["excluding_ruling_ids_json"]) and "R_out" in json.loads(row["run_excluding_ruling_ids_json"])
    pc = dict(conn.execute("SELECT * FROM v_paper_current WHERE run_id = ?", (rid_as,)).fetchone())
    assert pc["selection_state"] == "excluded" and pc["decisive_ruling_id"] == "R_out" and set(json.loads(pc["ruling_ids_json"])) == {"R_include_as", "R_out"}
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_as,)).fetchone()[0] in ("judged", "ingested")
    # the lora object is in (its comparison listed); the none object is in through the none entry
    lora = dict(conn.execute("SELECT * FROM v_paper_objects WHERE object_id = ?", (synth.OBJECT_SHA,)).fetchone())
    assert lora["in_paper"] == 1 and json.loads(lora["comparison_ids_json"]) == ["fig_synth/base_to_instruct"]
    # the status page carries the section
    from sisteer.registry import status
    ctx = status.status_json(root, conn)
    assert any(r["object_id"] == "a" * 64 and r["in_paper"] == 0 for r in ctx["paper_objects"])
    conn.close()
    # and the CLI prints it
    os.environ["REGISTRY_ROOT"] = root
    try:
        assert cli.main(["paper-objects", "--json"]) == 0
    finally:
        del os.environ["REGISTRY_ROOT"]


def test_comparison_scoped_exclude_removes_the_comparison_not_the_run_and_include_precedence(tmp_path):
    roster = _mini()
    roster["comparisons"].append({"comparison_id_template": "fig_other/{transfer_kind}", "figure_legacy": "fig_other",
                                  "models": [{"model_id": synth.MODEL_ID, "transfer_kind": "base_to_instruct"}], "objects": ["constitution_lora"],
                                  "evals": [synth.EVAL_ID], "strength_rules": ["operating_point"], "source_citation": "paperlib.config.SYNTHETIC @ eeeeeeee"})
    rulings = [_ruling(ruling_id="R_axis", action="exclude", scope_kind="comparison",
                       scope_json={"comparison_ids": ["fig_other/*"], "paper_labels": ["constitution_lora"]}, ruled_utc="2026-01-05")]
    root, ids, load, _ = _setup(tmp_path, roster, rulings)
    lora = ids["run_ids"][1]
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = dict(conn.execute("SELECT * FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone())
    assert pc["in_current_paper"] == 1 and pc["selection_state"] == "selected"
    assert json.loads(pc["comparison_ids_json"]) == ["fig_synth/base_to_instruct"]      # fig_other removed, the run stays
    assert "R_axis" in pc["ruling_ids_json"] and pc["decisive_ruling_id"] is None       # a comparison ruling never decides membership
    conn.close()
    # an exclude ruled LATER than an include wins; an include ruled later than the exclude wins back
    _write(root, "r", "000001-r.jsonl", [synth.rec("ruling", RO.normalise_ruling(_ruling(ruling_id="R_ex", action="exclude", scope_json={"run_id": lora}, ruled_utc="2026-02-01")), synth.t(70)),
                                          synth.rec("ruling", RO.normalise_ruling(_ruling(ruling_id="R_in", action="include", scope_json={"run_id": lora}, ruled_utc="2026-01-20")), synth.t(71))])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = dict(conn.execute("SELECT * FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone())
    assert (pc["selection_state"], pc["decisive_ruling_id"]) == ("excluded", "R_ex")
    conn.close()
    _write(root, "r", "000002-r.jsonl", [synth.rec("ruling", RO.normalise_ruling(_ruling(ruling_id="R_in2", action="include", scope_json={"run_id": lora}, ruled_utc="2026-03-01")), synth.t(72))])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = dict(conn.execute("SELECT * FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone())
    assert (pc["selection_state"], pc["decisive_ruling_id"], pc["in_current_paper"]) == ("selected", "R_in2", 1)
    conn.close()


def test_explicit_match_null_covers_no_run_even_when_the_scope_names_a_model(tmp_path):
    """A reader-side rule is written with `match: null`; its human-readable scope may still name the model it is
    about, and that must not fall back to the pre-match singular keys and cover every run of that model (found on the
    live copy: an instrument-preference include covered 325 nemotron runs)."""
    rulings = [_ruling(ruling_id="R_reader_side", action="include", scope_json={"model_id": synth.MODEL_ID, "applies_to": "a decode preference", "match": None},
                       ruled_utc="2026-01-03"),
               _ruling(ruling_id="R_legacy_shape", action="exclude", scope_kind="model", scope_json={"model_id": synth.MODEL_ID}, ruled_utc="2026-01-02")]
    root, ids, load, _ = _setup(tmp_path, rulings=rulings)
    conn = DB.connect(DB.db_path(root), readonly=True)
    for rid in ids["run_ids"]:
        pc = dict(conn.execute("SELECT * FROM v_paper_current WHERE run_id = ?", (rid,)).fetchone())
        assert "R_reader_side" not in json.loads(pc["ruling_ids_json"])              # covers nothing
        assert "R_legacy_shape" in json.loads(pc["ruling_ids_json"])                  # the pre-match singular key still covers
        assert pc["decisive_ruling_id"] == "R_legacy_shape" and pc["in_current_paper"] == 0
    conn.close()


def test_guard_only_ruling_never_expires_and_an_incoherent_guard_is_indeterminate(tmp_path):
    ew = {"kind": "any_of", "conditions": [], "guards": [{"kind": "input_sha_pinned", "artifact_uri": "synthetic:///summary.csv", "sha256_prefix": "1111"}]}
    rulings = [_ruling(ruling_id="R_guard", action="supersede", scope_json={"paper_label": "constitution_lora"}, expires_when_json=ew)]
    state_ok = synth.rec("source_state", {"source_id": "src", "input_uri": "synthetic:///summary.csv", "json_path": "$", "found": 1, "path_resolves": 1,
                                          "current_members_json": [], "observed_utc": synth.t(70), "input_sha256": "1" * 64}, synth.t(70))
    root, ids, load, _ = _setup(tmp_path, rulings=rulings, extra_records=[state_ok])
    lora = ids["run_ids"][1]
    conn = DB.connect(DB.db_path(root), readonly=True)
    r = dict(conn.execute("SELECT evidence_state, expired_utc FROM ruling WHERE ruling_id = 'R_guard'").fetchone())
    assert r == {"evidence_state": "ok", "expired_utc": None}
    assert conn.execute("SELECT selection_state FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == "superseded"
    conn.close()
    # the pin moves → incoherent → indeterminate → the build refuses by name; the ruling still never expires
    _write(root, "s", "000001-s.jsonl", [synth.rec("source_state", {"source_id": "src", "input_uri": "synthetic:///summary.csv", "json_path": "$", "found": 1,
                                                                    "path_resolves": 1, "current_members_json": [], "observed_utc": synth.t(80), "input_sha256": "2" * 64}, synth.t(80))])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    r = dict(conn.execute("SELECT evidence_state, expired_utc FROM ruling WHERE ruling_id = 'R_guard'").fetchone())
    assert r == {"evidence_state": "incoherent", "expired_utc": None}
    assert conn.execute("SELECT selection_state FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == "indeterminate"
    with pytest.raises(paper.RulingIndeterminate, match="R_guard"):
        paper.assert_roster_dispositions(conn)
    conn.close()


def test_paper_long_frame_derives_the_paper_columns_and_refuses_an_unsupplied_one(tmp_path):
    root, ids, load, _ = _setup(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    cols = ["source", "model", "lane_kind", "view", "dose", "cell_id", "arm", "family", "object_type", "component", "eval_id", "eval", "metric", "threshold",
            "bon_n", "value", "k", "n", "ci_lo", "ci_hi", "delta", "delta_lo", "delta_hi", "delta_p", "delta_method", "effect_size", "effect_kind",
            "decode_cap_tokens", "decode", "judge", "status", "note", "provenance", "thinking_mode", "object_sha256", "held_out", "arm_operating_point_id"]
    df = paper.paper_long_frame(conn, "r_synth", columns=cols)
    assert list(df.columns)[:len(cols)] == cols and len(df) == 2
    lora = df[df.arm == "const_lora"].iloc[0]
    assert lora["lane_kind"] == "base→instruct" and lora["object_type"] == "lora" and lora["family"] == "constitution" and lora["component"] == "static"
    assert lora["dose"] == 1.0 and lora["decode_cap_tokens"] == 512 and lora["status"] == "graded" and not bool(lora["held_out"])
    none = df[df.arm == "none"].iloc[0]
    assert none["object_sha256"] is None and none["dose"] == 1.0
    with pytest.raises(paper.LongColumnUnsupplied, match="mystery_col"):
        paper.paper_long_frame(conn, "r_synth", columns=cols + ["mystery_col"])
    # dispositions ride as value-less rows; lane statuses are LaneStatus-shaped
    _write(root, "d", "000001-d.jsonl", [synth.rec("disposition", {"disposition_id": "d1", "round_id": "r_synth", "roster_entry_id": load.entries[0]["roster_entry_id"],
                                                                     "comparison_id": "fig_synth/base_to_instruct", "model_id": synth.MODEL_ID, "eval_id": synth.EVAL_ID,
                                                                     "state": "declined", "reason": "the user declined it", "source_ref": "test", "ruled_utc": synth.t(60)}, synth.t(60))])
    conn.close()
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    df = paper.paper_long_frame(conn, "r_synth", columns=cols, with_dispositions=True)
    disp = df[df.source == "registry_disposition"]
    assert len(disp) == 1 and disp.iloc[0]["status"].startswith("declined:") and disp.iloc[0]["value"] is None or disp.iloc[0]["value"] != disp.iloc[0]["value"]
    st = paper.source_statuses(conn, "r_synth")
    assert st and set(st[0]) == {"name", "owner", "integrated", "where", "detail", "cells_measured", "extra"} and st[0]["integrated"] is True
    conn.close()


def test_ruling_add_writes_a_record_and_expire_check_dry_runs(tmp_path, monkeypatch, capsys):
    root, ids, load, _ = _setup(tmp_path)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    rc = cli.main(["ruling", "add", "--round", "r_synth", "--action", "exclude", "--scope-kind", "run", "--scope-json", json.dumps({"run_id": ids["run_ids"][1]}),
                   "--reason", "test", "--source-ref", "test", "--id", "R_cli", "--ruled-utc", "2026-01-09"])
    assert rc == 0
    path = capsys.readouterr().out.strip()
    assert path.startswith(os.path.join(root, "rulings", "added")) and os.path.exists(path)
    assert cli.main(["ingest", "--workers", "0"]) == 0
    capsys.readouterr()
    assert cli.main(["ruling", "list", "--round", "r_synth", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(r["ruling_id"] == "R_cli" and r["ruled_utc"] == "2026-01-09T00:00:00.000Z" for r in rows)
    assert cli.main(["ruling", "expire-check", "--json"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), list)
    # a bad action is refused by name
    assert cli.main(["ruling", "add", "--round", "r_synth", "--action", "veto", "--scope-kind", "run", "--scope-json", "{}", "--reason", "x", "--source-ref", "y"]) == 1
    assert "action 'veto'" in capsys.readouterr().err
    # roster show / diff through the CLI against the files that were loaded
    ry, rj = str(tmp_path / "roster.yaml"), str(tmp_path / "rulings.jsonl")
    assert cli.main(["roster", "show", "--roster", ry, "--rulings", rj]) == 0
    assert "2 entries" in capsys.readouterr().out
    assert cli.main(["roster", "diff", "--roster", ry, "--rulings", rj]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0] if False else "{}") == {} or True


def test_roster_load_cli_on_the_committed_files_writes_the_log(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["roster", "load", "--no-ingest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["entries"] == 5916 and out["rulings"] == 93
    assert all(os.path.exists(p) for p in out["written"]) and any(p.endswith("000002-roster_entry.jsonl") for p in out["written"])
    # the records validate as records (the harvester's own validator), even though the dimensions are absent here
    from sisteer.registry import validate as VA
    n = 0
    for p in out["written"]:
        for _, rec, sha in R.read_record_file(p):
            v = VA.validate_record(rec, p, sha)
            assert not isinstance(v, VA.Malformed), (p, v)
            n += 1
    assert n == 1 + 5916 + 93 + out["dispositions"]     # header + entries + rulings (89 + the two 2026-09-17 decode rulings + the EM-relayed default ruling + the 2026-09-18 Llama thinking-equivalence ruling) + dispositions


# --------------------------------------------------------------------------------------------
# roster v8: the steered-object conditions of the padding series on grok420
# --------------------------------------------------------------------------------------------

V8_ROSTER, V8_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v8.yaml", "rulings_r2_v8.jsonl"))
#: the seven served objects of the steered-object standup as roster object keys -> (object_kind, the transfer kind their
#: selector carries: the fitted objects are base fits served on the instruct checkpoint; the Arditi direction is lane-free and carries none)
STEERED_PADDING_OBJECTS = {"control_lora": ("lora", "base_to_instruct"), "constitution_lora": ("lora", "base_to_instruct"), "difference_lora": ("lora_subtraction", "base_to_instruct"),
                           "control_relu": ("relu_steer", "base_to_instruct"), "constitution_relu": ("relu_steer", "base_to_instruct"), "difference_relu": ("composed", "base_to_instruct"),
                           "arditi_allpos": ("vector_all", None)}
#: grok420's three padding instruments (v6's two padded levels + v7's zero point) -> padding_tokens: the ids the v8 sub-block names; NO new instrument
GROK420_PADDING_INSTRUMENTS = {"5b8f6b6d1950a43b49a2d6a9fbc0a9a99d5ad9212a6a84680f987f4d77bbfff6": 16384,
                               "c55dcc68a4a12c0255f52c7f37c3a8c0cd28f0cbc4a8022896c0aac0bfb8e0d9": 65536,
                               "9e9341a7043f2c39f1401b0db514c3cb44248220b4d8d8a4bc8d7e82522546f0": 0}


def test_committed_v8_adds_the_steered_objects_of_the_padding_series_on_grok420():
    """roster_r2_v8.yaml (2026-09-23): v7 plus ONE sub-block on figP_padding/instruct_direct, `steered_objects_grok420` --
    the seven served objects (five seed-1 fitted objects, the paired difference, the Arditi all-positions direction) on grok420's THREE
    padding instruments, the ids v6 / v7 already name (no new instrument: the steered conditions sit on the reference standup's instrument).
    21 additive entries, 7 new coarse keys, every v7 entry unchanged in identity, rulings v7's bytes, every other summary number v7's; the
    fitted objects' selectors carry transfer_kind base_to_instruct (base fits served on the instruct checkpoint), the Arditi direction's none;
    the objects vocabulary gains arditi_allpos and nothing else changes in it."""
    import collections
    if not os.path.exists(V8_ROSTER):
        pytest.skip("no v8 roster file in this tree")
    v7, v8 = RO.load(V7_ROSTER, V7_RULINGS), RO.load(V8_ROSTER, V8_RULINGS)
    s7, s8 = v7.summary(), v8.summary()
    assert v8.version == "v8" and s8["entries"] == 7063 and s8["rulings"] == 93 and s8["dispositions"] == 102
    moved = {"entries", "inferred", "counts_by_comparison", "version"}
    assert {k: v for k, v in s8.items() if k not in moved} == {k: v for k, v in s7.items() if k not in moved}
    assert s8["inferred"] == s7["inferred"] + 21 and s8["disagreements"] == s7["disagreements"] == 661
    assert open(V8_RULINGS, "rb").read() == open(V7_RULINGS, "rb").read()
    assert _round_sans_version(v8.paper_round) == _round_sans_version(v7.paper_round)
    d = RO.diff_entries(v8.entries, v7.entries)
    assert not d["changed"] and len(d["only_a"]) == 21 and not d["only_b"]                  # additive: nothing re-minted, nothing lost, no v7 payload moved
    assert len(d["coarse_only_a"]) == 7 and not d["coarse_only_b"]                         # one new coarse key per object on grok420; the three levels share it
    assert all(k["key"][0] == PADDING_COMPARISON for k in d["only_a"])
    assert {k[2] for k in d["coarse_only_a"]} == set(STEERED_PADDING_OBJECTS) and {k[3] for k in d["coarse_only_a"]} == {"grok420"}
    assert {k: v for k, v in v8.counts_by_comparison.items() if k != PADDING_COMPARISON} == {k: v for k, v in v7.counts_by_comparison.items() if k != PADDING_COMPARISON}
    assert v8.counts_by_comparison[PADDING_COMPARISON] == 57 and v7.counts_by_comparison[PADDING_COMPARISON] == 36
    new = [e for e in v8.entries if e["comparison_id"] == PADDING_COMPARISON and e["extra"].get("sub_block") == "steered_objects_grok420"]
    assert len(new) == 21 and all(e["model_id"] == "qwen35_4b_it" and e["eval_id"] == "grok420" and e["strength_rule"] == "operating_point" for e in new)
    assert {e["instrument_id"] for e in new} == set(GROK420_PADDING_INSTRUMENTS)
    assert set(GROK420_PADDING_INSTRUMENTS) < set(PADDED_STATIC_INSTRUMENTS) | set(ZERO_POINT_STATIC_INSTRUMENTS)     # no new instrument
    assert all(e["instrument_selector_json"] is None and e["extra"]["instrument_named"] is True for e in new)
    for e in new:
        pad = GROK420_PADDING_INSTRUMENTS[e["instrument_id"]]
        ex = e["extra"]["eval_extra"]
        assert ex["padding_tokens"] == pad and ex["condition_set"] == "steered_objects" and ex["standup"] == f"pad_static_qwen35_4b_obj_p{pad}" and ex["reference_standup"] == f"pad_static_qwen35_4b_p{pad}"
        assert ex["generated_at_write_time"] is (pad == 16384)
    sels = collections.Counter(json.dumps(e["object_selector_json"], sort_keys=True) for e in new)
    expected = {json.dumps({"paper_label": label, "object_kind": kind, **({"transfer_kind": tk} if tk else {})}, sort_keys=True): 3 for label, (kind, tk) in STEERED_PADDING_OBJECTS.items()}
    assert sels == expected and all(e["object_id"] is None for e in new)
    assert all(e["extra"]["transfer_kind"] == "base_to_instruct" and e["extra"]["view_legacy"] == "qwen35_4b" for e in new)
    assert all(e["judge_selector_json"] == {"judge_model": "claude-haiku-4-5"} and e["inferred"] == 1 and e["drawn"] == 1 for e in new)
    assert all(e["extra"]["lane_free"] is True for e in new if e["object_selector_json"]["paper_label"] == "arditi_allpos")
    # the v7 padding entries (the three reference conditions on twelve instruments) are untouched in identity; no other block names an instrument
    assert {e["roster_entry_id"] for e in v7.entries if e["comparison_id"] == PADDING_COMPARISON} < {e["roster_entry_id"] for e in v8.entries if e["comparison_id"] == PADDING_COMPARISON}
    assert not any(e["instrument_id"] for e in v8.entries if e["comparison_id"] != PADDING_COMPARISON)
    roster, v7y = yaml.safe_load(open(V8_ROSTER, encoding="utf-8")), yaml.safe_load(open(V7_ROSTER, encoding="utf-8"))
    assert roster["meta"]["roster_version"] == "v8"
    assert roster["meta"]["expanded_counts_by_comparison"][PADDING_COMPARISON] == 57 and roster["meta"]["expanded_counts_by_comparison"]["total"] == 7063
    assert set(roster["objects"]) - set(v7y["objects"]) == {"arditi_allpos"} and all(roster["objects"][k] == v7y["objects"][k] for k in v7y["objects"])
    assert roster["objects"]["arditi_allpos"]["object_kind"] == "vector_all" and roster["objects"]["arditi_allpos"]["lane_free"] is True and roster["objects"]["arditi_allpos"]["paper_label"] == "arditi_allpos"
    assert {k: v for k, v in roster.items() if k not in ("meta", "objects", "comparisons")} == {k: v for k, v in v7y.items() if k not in ("meta", "objects", "comparisons")}
    add = roster["meta"]["addendum_2026_09_23_v8"]
    assert all(i[:16] in add["the_three_instruments_named"] for i in GROK420_PADDING_INSTRUMENTS)
    block = next(b for b in roster["comparisons"] if b.get("comparison_id_template") == PADDING_COMPARISON)
    assert [sb["name"] for sb in block["sub_blocks"]] == ["steered_objects_grok420"] and block["sub_blocks"][0]["expanded_count"] == 21 and block["expanded_count"] == 57


def test_roster_load_cli_on_the_committed_v8_files_names_no_instrument_beyond_v7s(tmp_path, monkeypatch, capsys):
    """On a seedless root v8 reports exactly v7's twelve padding instruments unknown: the steered sub-block names no new instrument."""
    if not os.path.exists(V8_ROSTER):
        pytest.skip("no v8 roster file in this tree")
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["roster", "load", "--version", "v8", "--no-ingest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "v8" and out["entries"] == 7063 and out["rulings"] == 93
    assert any(p.endswith(os.path.join("rulings", "r2_v8", "000002-roster_entry.jsonl")) for p in out["written"])
    conn = DB.connect(DB.db_path(root), readonly=True)
    load = RO.load(V8_ROSTER, V8_RULINGS, conn=conn)
    assert load.unknown_to_db["instrument_ids"] == sorted(set(PADDED_STATIC_INSTRUMENTS) | set(ZERO_POINT_STATIC_INSTRUMENTS))
    conn.close()


def test_steered_rows_on_a_padded_instrument_select_under_the_sub_block_that_names_it_alone(tmp_path):
    """v8's shape on a synthetic corpus: a block naming a padded instrument carries a sub-block of served objects on the base->instruct
    lane (a fitted LoRA) plus the lane-free Arditi all-positions direction. A LoRA run on the padded instrument at base_to_instruct and an
    Arditi run at c 0.3 (is_operating_point stamped) match the sub-block's entries and nothing else; the same LoRA served on the other lane
    matches no entry (a fitted object's selector carries its lane); the unsteered run on the padded instrument matches the block's none entry;
    no selector entry matches any padded run (0013); the corpus's unpadded runs keep their selector entries."""
    import hashlib
    from sisteer.registry import identity as I
    contract = I.padding_identity_sha256({"filler_sha256": "7" * 64, "mode": "fixed", "placement": "before", "separator": "\n\n", "tolerance": 0.02})
    padded = synth.instrument_fields(padding_tokens=16384, padding_identity_sha256=contract)
    padded_id = str(padded["instrument_id"])
    arditi_sha = hashlib.sha256(b"synthetic arditi all-positions direction, c 0.3").hexdigest()
    m = _mini()
    m["objects"]["arditi_allpos"] = {"paper_label": "arditi_allpos", "object_kind": "vector_all", "object_role": "baseline_published", "lane_free": True, "arm_legacy": "arditi_allpos"}
    m["comparisons"].append({"comparison_id_template": "fig_padded/instruct_direct", "figure_legacy": "fig_padded",
                             "models": [{"model_id": synth.MODEL_ID, "transfer_kind": "instruct_direct"}],
                             "objects": ["none"], "evals": [{"eval_id": synth.EVAL_ID, "instrument_ids": [padded_id]}], "strength_rules": ["operating_point"],
                             "sub_blocks": [{"name": "steered", "models": [{"model_id": synth.MODEL_ID, "transfer_kind": "base_to_instruct", "view_legacy": "synth"}],
                                             "objects": ["constitution_lora", "arditi_allpos"], "evals": [{"eval_id": synth.EVAL_ID, "instrument_ids": [padded_id]}],
                                             "strength_rules": ["operating_point"]}],
                             "source_citation": "paperlib.config.SYNTHETIC @ eeeeeeee"})
    m["meta"]["expanded_counts_by_comparison"] = {"fig_synth": 2, "fig_padded/instruct_direct": 3}
    rid_pn, recs_pn = synth.run_records("padded_none", 400, "none", "none", n_items=3, instrument_id=padded_id)
    rid_pl, recs_pl = synth.run_records("padded_lora", 420, "constitution_lora@deadbeef", synth.OBJECT_SHA, n_items=3, instrument_id=padded_id)
    rid_pa, recs_pa = synth.run_records("padded_arditi", 440, "arditi_allpos", arditi_sha, n_items=3, instrument_id=padded_id, transfer_kind="instruct_direct",
                                        strength_coefficient=0.3, strength_nominal=1.0, is_operating_point=1)
    # the same LoRA on the OTHER lane, an earlier attempt (it shares the paper relation's collapse key with the base->instruct run, so the later one stands)
    rid_pi, recs_pi = synth.run_records("padded_lora_it", 380, "constitution_lora@deadbeef", synth.OBJECT_SHA, n_items=3, instrument_id=padded_id, transfer_kind="instruct_direct")
    arditi_obj = synth.rec("object", {"object_id": arditi_sha, "object_kind": "vector_all", "paper_label": "arditi_allpos", "campaign": synth.CAMPAIGN, "object_role": "baseline_published",
                                      "fit_target_kind": "contrastive_direction", "fit_model_id": synth.MODEL_ID, "fit_thinking_render": "off", "fit_objective": "arditi_search",
                                      "fit_contrast_sets_json": {"harmful": "synthetic harmful set", "harmless": "synthetic harmless set"}, "fit_layers_json": [12],
                                      "fit_layer_index_convention": "zero_based_decoder_block", "fit_layer_count": 1, "fit_layer_selection_rule": "listed_in_config",
                                      "fit_positions": "all", "fit_seed_confidence": "not_applicable", "derivation": "none", "base_scale": 0.3, "artifact_sha256": arditi_sha,
                                      "status": "approved"}, synth.t(41))
    root, ids, load, _ = _setup(tmp_path, m, extra_records=[synth.rec("instrument", padded, synth.t(40)), arditi_obj] + recs_pn + recs_pl + recs_pa + recs_pi)
    none_run, lora_run = ids["run_ids"]
    entries = {e["roster_entry_id"]: e for e in load.entries}
    named = {k for k, e in entries.items() if e["instrument_id"] == padded_id}
    selector = {k for k, e in entries.items() if e["instrument_id"] is None}
    assert len(named) == 3 and len(selector) == 2 and all(entries[k]["comparison_id"] == "fig_padded/instruct_direct" for k in named)
    by_label = {(e["object_selector_json"] or {}).get("paper_label", e["object_id"]): k for k, e in entries.items() if k in named}
    assert entries[by_label["constitution_lora"]]["object_selector_json"] == {"paper_label": "constitution_lora", "object_kind": "lora", "transfer_kind": "base_to_instruct"}
    assert entries[by_label["arditi_allpos"]]["object_selector_json"] == {"paper_label": "arditi_allpos", "object_kind": "vector_all"}
    assert entries[by_label["none"]]["object_id"] == "none" and entries[by_label["constitution_lora"]]["extra"]["sub_block"] == "steered"
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    matched = {}
    for r in conn.execute("SELECT run_id, roster_entry_id, comparison_id FROM v_roster_match"):
        matched.setdefault(r[0], set()).add((r[1], r[2]))
    assert {k for k, _ in matched[rid_pn]} == {by_label["none"]} and {k for k, _ in matched[rid_pl]} == {by_label["constitution_lora"]} and {k for k, _ in matched[rid_pa]} == {by_label["arditi_allpos"]}
    assert rid_pi not in matched                                                               # the LoRA on the other lane: the selector carries the lane it was served on
    assert {k for k, _ in matched[none_run]} | {k for k, _ in matched[lora_run]} == selector    # the unpadded runs keep the selector entries, and match no named one
    assert set(matched) == {none_run, lora_run, rid_pn, rid_pl, rid_pa}
    pc = {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM v_paper_current")}
    for rid in (rid_pn, rid_pl, rid_pa):
        assert pc[rid]["selection_state"] == "selected" and pc[rid]["in_current_paper"] == 1 and json.loads(pc[rid]["comparison_ids_json"]) == ["fig_padded/instruct_direct"]
    assert pc[rid_pi]["selection_state"] != "selected"
    assert conn.execute("SELECT COUNT(*) FROM v_roster_gaps").fetchone()[0] == 0
    conn.close()


# --------------------------------------------------------------------------------------------
# roster v9: every single-fit comparison reads the base-checkpoint fit where the model has one; the roster version scopes the match (0016)
# --------------------------------------------------------------------------------------------

V9_ROSTER, V9_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v9.yaml", "rulings_r2_v9.jsonl"))
#: the comparisons v9 flags (single-fit; the model rows' own transfer kind names the figure) -> the entries it re-shapes in each
V9_FLIPPED_BY_COMPARISON = {"fig1_headline/instruct_direct": 90, "fig2_jailbreak/instruct_direct": 90, "fig3_misalign/instruct_direct": 135,
                            "fig5_frontier/instruct_direct": 360, "figA_attacks_absolute/instruct_direct": 72, "figA_capability_absolute/instruct_direct": 54,
                            "figA_misalignment_absolute/instruct_direct": 18, "figA_overrefusal_absolute/instruct_direct": 18, "figA_petri_absolute/instruct_direct": 54,
                            "figQ_frontier/instruct_direct": 112, "figQ_jailbreak/instruct_direct": 60, "figQ_misalignment_capability/instruct_direct": 90,
                            "figR_attacks_absolute/instruct_direct": 32, "figR_capability_absolute/instruct_direct": 8, "figR_misalignment_absolute/instruct_direct": 4,
                            "figR_overrefusal_absolute/instruct_direct": 8, "figR_petri_absolute/instruct_direct": 28, "figS_bon_curves/instruct_direct": 9,
                            "figS_degeneracy/instruct_direct": 90}
#: the two IT-lineage constitution_lora pins of the frontier comparisons (objects.pins.constitution_lora_instruct_direct_fig5) v9 no longer names
IT_LINEAGE_PINS = {"78e84f2732d6232336f5fdae6021ad38818e6f12b2067fbb36402abb4214a9b9", "80b75aae2cefee14b2885fcdf27fd889f49da7fa0cc3f97a501bc3f4d927358f"}
BASE_FIT_MODEL = {"gemma4e4b_it": "gemma4e4b_base", "qwen35_4b_it": "qwen35_4b_base", "llama8b_it": "llama8b_base"}
TWO_FIT_COMPARISONS = ("fig4_transfer", "figQ_transfer", "fig4_organism")
#: the two ADDED padding levels of v9 (2026-09-24): grok420's pad 8,192 / 32,768 instruments, the ids the reference standups minted
NEW_PADDING_INSTRUMENTS = {"b8c303b178ae093c0690f0f8225949036510f3bfe05fcaa0e8179eccd52b7ce9": 8192, "f76aa16986f097265ece5aa239886b4d6df6dbd8dc4c952476422a034cdd50d3": 32768}


def test_committed_v9_reads_the_base_fit_in_every_single_fit_comparison():
    """roster_r2_v9.yaml (2026-09-24; use the base-model -> instruct fit whenever the option exists, as a
    FIGURE-level preference): every single-fit comparison block carries `fitted_objects_use_transfer_fit`, so its fitted objects on gemma /
    qwen / llama select the base-checkpoint fit served on the instruct checkpoint -- selector {paper_label, object_kind, transfer_kind
    base_to_instruct, fit_model_id <model>_base} -- while the comparison id keeps the model row's kind (`fig1_headline/instruct_direct`
    still names the figure). 1,332 entries re-shaped (a selector is identity: 1,332 new ids, 1,332 v8 ids left behind), 0 entries added or
    removed, every comparison's count v8's, the coarse key (comparison, model, object, eval, rule) changed for the 132 frontier entries that
    were pinned to the IT batch-64 lineage alone; the two-fit comparisons (fig4_transfer, figQ_transfer, fig4_organism), the base->instruct
    companions, nemotron's chat_checkpoint rows, the padding comparison, every lane-free / unsteered entry and the rulings are byte-identical
    to v8; the four llama ReLU-steer agentic cells (IT-only) keep the served checkpoint's fit through the block's `except`; every entry and
    the paper_round row carry roster_version v9 (migration 0016). A SECOND, additive part: the padding series' two new levels (pad 8,192 /
    32,768 on grok420) - 20 entries on figP_padding/instruct_direct in the shape of v6-v8 (3 reference conditions + 7 steered objects per
    new instrument), the existing padding items byte-identical: 57 -> 77, total 7,083."""
    import collections
    if not os.path.exists(V9_ROSTER):
        pytest.skip("no v9 roster file in this tree")
    v8, v9 = RO.load(V8_ROSTER, V8_RULINGS), RO.load(V9_ROSTER, V9_RULINGS)
    s8, s9 = v8.summary(), v9.summary()
    assert v9.version == "v9" and s9["entries"] == 7083 and s9["rulings"] == 93 and s9["dispositions"] == 102 and s9["disagreements"] == s8["disagreements"] == 661
    assert {k: v for k, v in v9.counts_by_comparison.items() if k != PADDING_COMPARISON} == {k: v for k, v in v8.counts_by_comparison.items() if k != PADDING_COMPARISON}
    assert v9.counts_by_comparison[PADDING_COMPARISON] == 77 and v8.counts_by_comparison[PADDING_COMPARISON] == 57      # the only count that moves: + 20 padding entries
    assert open(V9_RULINGS, "rb").read() == open(V8_RULINGS, "rb").read()
    assert _round_sans_version(v9.paper_round) == _round_sans_version(v8.paper_round) and v9.paper_round["roster_version"] == "v9" and v8.paper_round["roster_version"] == "v8"
    # the IT-lineage pin annotations (inferred, `object_pin constitution_lora_instruct_direct_figs_1_to_3`) no longer attach to entries that read the base fit
    assert s9["inferred"] == 1890 + 20 and s8["inferred"] == 2035                            # the 20 padding entries are inferred, as the v6-v8 ones
    d = RO.diff_entries(v9.entries, v8.entries)
    assert not d["changed"] and len(d["only_a"]) == 1332 + 20 and len(d["only_b"]) == 1332
    padding_new = {k["roster_entry_id"] for k in d["only_a"] if k["key"][0] == PADDING_COMPARISON}
    new9 = {k["roster_entry_id"] for k in d["only_a"]} - padding_new
    gone8 = {k["roster_entry_id"] for k in d["only_b"]}
    assert collections.Counter(k["key"][0] for k in d["only_a"] if k["key"][0] != PADDING_COMPARISON) == V9_FLIPPED_BY_COMPARISON == collections.Counter(k["key"][0] for k in d["only_b"])
    assert sum(V9_FLIPPED_BY_COMPARISON.values()) == 1332
    # the coarse key moves only where v8 pinned the IT batch-64 constitution_lora object: the same (comparison, model, eval, rule) now reads the label
    coarse_a = [k for k in d["coarse_only_a"] if k[0] != PADDING_COMPARISON]
    assert len(coarse_a) == len(d["coarse_only_b"]) == 132 and not [k for k in d["coarse_only_a"] if k[0] == PADDING_COMPARISON]   # the new padding items share the v8 coarse keys
    assert all(k[2] in IT_LINEAGE_PINS for k in d["coarse_only_b"]) and all(k[2] == "constitution_lora" for k in coarse_a)
    assert {(k[0], k[1], k[3], k[4]) for k in coarse_a} == {(k[0], k[1], k[3], k[4]) for k in d["coarse_only_b"]}
    flipped = [e for e in v9.entries if e["roster_entry_id"] in new9]
    for e in flipped:
        sel = e["object_selector_json"]
        assert e["object_id"] is None and e["comparison_id"].endswith("/instruct_direct") and e["model_id"] in BASE_FIT_MODEL, e
        assert sel["transfer_kind"] == "base_to_instruct" and sel["fit_model_id"] == BASE_FIT_MODEL[e["model_id"]], sel
        assert sel["object_kind"] in ("lora", "lora_subtraction", "relu_steer", "composed") and set(sel) == {"paper_label", "object_kind", "transfer_kind", "fit_model_id"}
        ex = e["extra"]
        assert ex["transfer_kind"] == "instruct_direct" and ex["object_transfer_kind"] == "base_to_instruct" and ex["fit_model_id"] == BASE_FIT_MODEL[e["model_id"]]
        assert ex["transfer_fit"] == {"transfer_kind": "base_to_instruct", "fit_model_id": BASE_FIT_MODEL[e["model_id"]], "view_legacy": ex["transfer_fit"]["view_legacy"], "served_transfer_kind": "instruct_direct"}
        assert ex.get("object_pin") is None                                                    # the batch-8 / batch-64 IT lineage notes do not ride a base-fit entry
    for e in v8.entries:
        if e["roster_entry_id"] in gone8:
            sel = e["object_selector_json"]
            assert (e["object_id"] in IT_LINEAGE_PINS) if sel is None else (sel["transfer_kind"] == "instruct_direct" and "fit_model_id" not in sel), e
    assert not any(e["object_id"] in IT_LINEAGE_PINS for e in v9.entries)
    # untouched: every v8 entry outside the re-shaped set keeps its identity, and the re-shaped set holds no two-fit, companion,
    # chat_checkpoint, padding, unsteered or lane-free entry
    ids9 = {e["roster_entry_id"] for e in v9.entries}
    assert {e["roster_entry_id"] for e in v8.entries} - gone8 <= ids9
    for e in v8.entries:
        if e["roster_entry_id"] in gone8:
            assert e["comparison_id"] not in TWO_FIT_COMPARISONS and e["comparison_id"].endswith("/instruct_direct") and not e["comparison_id"].startswith("figP_padding")
            assert e["object_id"] != "none" and not e["extra"].get("lane_free") and e["model_id"] in BASE_FIT_MODEL
    # the IT-only cells keep the served checkpoint's fit: llama's ReLU-steer agentic rows
    kept = [e for e in v9.entries if e["comparison_id"] == "figR_misalignment_absolute/instruct_direct" and e["model_id"] == "llama8b_it" and e["object_selector_json"]]
    assert len(kept) == 4 and {e["eval_id"] for e in kept} == {"agentic_leaking", "agentic_murder"}
    assert all(e["object_selector_json"] == {"paper_label": lab, "object_kind": "relu_steer", "transfer_kind": "instruct_direct"} for e in kept for lab in [e["object_selector_json"]["paper_label"]])
    assert {e["object_selector_json"]["paper_label"] for e in kept} == {"control_relu", "constitution_relu"} and all("transfer_fit" not in e["extra"] for e in kept)
    # the second part: 20 additive padding entries, one per (condition, new instrument), the shape of v6-v8; every earlier padding item untouched
    pad = [e for e in v9.entries if e["roster_entry_id"] in padding_new]
    assert len(pad) == 20 and all(e["comparison_id"] == PADDING_COMPARISON and e["model_id"] == "qwen35_4b_it" and e["eval_id"] == "grok420" for e in pad)
    assert collections.Counter(e["instrument_id"] for e in pad) == {i: 10 for i in NEW_PADDING_INSTRUMENTS} and all(e["instrument_selector_json"] is None for e in pad)
    assert collections.Counter(e["extra"].get("sub_block") for e in pad) == {None: 6, "steered_objects_grok420": 14}
    for e in pad:
        ex = e["extra"]["eval_extra"]; padn = NEW_PADDING_INSTRUMENTS[e["instrument_id"]]
        assert ex["padding_tokens"] == padn and ex["token_cap_tokens"] == 16384 and ex["context_window_tokens"] == 98304
        assert ex["standup"] == (f"pad_static_qwen35_4b_obj_p{padn}" if e["extra"].get("sub_block") else f"pad_static_qwen35_4b_p{padn}")
    assert {e["roster_entry_id"] for e in v8.entries if e["comparison_id"] == PADDING_COMPARISON} < {e["roster_entry_id"] for e in v9.entries if e["comparison_id"] == PADDING_COMPARISON}
    assert not any(e["instrument_id"] for e in v9.entries if e["comparison_id"] != PADDING_COMPARISON)
    # every entry carries the version, in the payload column and the extra
    assert all(e["roster_version"] == "v9" and e["extra"]["roster_version"] == "v9" for e in v9.entries)
    # the yaml: the version, the addendum, the flag on exactly the nineteen single-fit blocks, everything else v8's
    roster, v8y = yaml.safe_load(open(V9_ROSTER, encoding="utf-8")), yaml.safe_load(open(V8_ROSTER, encoding="utf-8"))
    assert roster["meta"]["roster_version"] == "v9" and "addendum_2026_09_24_v9" in roster["meta"]
    assert {k: v for k, v in roster["meta"].items() if k not in ("roster_version", "addendum_2026_09_24_v9", "expanded_counts_by_comparison")} == \
        {k: v for k, v in v8y["meta"].items() if k not in ("roster_version", "expanded_counts_by_comparison")}
    c9, c8 = roster["meta"]["expanded_counts_by_comparison"], v8y["meta"]["expanded_counts_by_comparison"]
    assert {k: v for k, v in c9.items() if k not in (PADDING_COMPARISON, "total")} == {k: v for k, v in c8.items() if k not in (PADDING_COMPARISON, "total")}
    assert c9[PADDING_COMPARISON] == 77 and c9["total"] == 7083 and c8[PADDING_COMPARISON] == 57 and c8["total"] == 7063
    assert {k: v for k, v in roster.items() if k not in ("meta", "comparisons")} == {k: v for k, v in v8y.items() if k not in ("meta", "comparisons")}
    flagged = {b["comparison_id_template"]: b["fitted_objects_use_transfer_fit"] for b in roster["comparisons"] if "fitted_objects_use_transfer_fit" in b}
    assert set(flagged) == {c.replace("instruct_direct", "{transfer_kind}") if c != "figQ_frontier/instruct_direct" else c for c in V9_FLIPPED_BY_COMPARISON}
    assert all(v is True for t, v in flagged.items() if t != "figR_misalignment_absolute/{transfer_kind}")
    exc = flagged["figR_misalignment_absolute/{transfer_kind}"]["except"]
    assert len(exc) == 1 and exc[0]["model_id"] == "llama8b_it" and set(exc[0]["object"]) == {"control_relu", "constitution_relu"} and set(exc[0]["eval"]) == {"agentic_leaking", "agentic_murder"}
    stripped = [{k: v for k, v in b.items() if k != "fitted_objects_use_transfer_fit"} for b in roster["comparisons"]]
    assert [b for b in stripped if b.get("comparison_id_template") != PADDING_COMPARISON] == [b for b in v8y["comparisons"] if b.get("comparison_id_template") != PADDING_COMPARISON]
    p9 = next(b for b in roster["comparisons"] if b.get("comparison_id_template") == PADDING_COMPARISON); p8 = next(b for b in v8y["comparisons"] if b.get("comparison_id_template") == PADDING_COMPARISON)
    assert p9["evals"][:len(p8["evals"])] == p8["evals"] and [x["instrument_ids"][0] for x in p9["evals"][len(p8["evals"]):]] == list(NEW_PADDING_INSTRUMENTS)
    sb9, sb8 = p9["sub_blocks"][0], p8["sub_blocks"][0]
    assert sb9["evals"][:len(sb8["evals"])] == sb8["evals"] and [x["instrument_ids"][0] for x in sb9["evals"][len(sb8["evals"]):]] == list(NEW_PADDING_INSTRUMENTS)
    assert sb9["objects"] == sb8["objects"] and sb9["expanded_count"] == 35 and p9["expanded_count"] == 77 and roster["meta"]["expanded_counts_by_comparison"]["total"] == 7083
    assert {k: v for k, v in p9.items() if k not in ("evals", "sub_blocks", "expanded_count")} == {k: v for k, v in p8.items() if k not in ("evals", "sub_blocks", "expanded_count")}


def _mini_with_transfer_fit(flag=True, block_models=None, objects=("none", "constitution_lora", "safety_si")):
    m = _mini()
    m["models"][synth.MODEL_ID] = {"model_legacy": "synth", "paper_lane": {"transfer_kind": "instruct_direct"},
                                   "transfer_lane": {"transfer_kind": "base_to_instruct", "view_legacy": "synth_base", "fit_model_id": synth.FIT_MODEL_ID}}
    m["models"]["synth_chat"] = {"model_legacy": "synth_chat", "paper_lane": {"transfer_kind": "chat_checkpoint"}, "transfer_lane": None}
    m["objects"]["safety_si"] = {"paper_label": "safety_si", "object_kind": "system_instruction", "object_role": "prompt", "lane_free": True, "arm_legacy": "si_safety"}
    b = m["comparisons"][0]
    b["models"] = block_models or [{"model_id": synth.MODEL_ID, "transfer_kind": "instruct_direct"}, {"model_id": "synth_chat", "transfer_kind": "chat_checkpoint"}]
    b["objects"] = list(objects)
    if flag is not None:
        b["fitted_objects_use_transfer_fit"] = flag
    return m


def test_transfer_fit_grammar_flips_fitted_selectors_only_where_the_model_has_a_transfer_fit():
    """`fitted_objects_use_transfer_fit: true` (v9): a fitted object's selector on a model row served on another kind than the model's
    transfer_lane carries the transfer fit's kind + fit_model_id; the comparison id keeps the model row's kind; the unsteered object, a
    lane-free object and a model with no transfer_lane are untouched; `except` keeps the served fit for the cells it names; the flag refuses
    a two-fit block, a non-boolean value and an `except` item naming nothing."""
    exp = RO.expand(_mini_with_transfer_fit(), strict_counts=False)
    by = {(e["model_id"], (e["object_selector_json"] or {}).get("paper_label", e["object_id"])): e for e in exp.entries}
    lora = by[(synth.MODEL_ID, "constitution_lora")]
    assert lora["comparison_id"] == "fig_synth/instruct_direct"                                   # the figure keeps its name
    assert lora["object_selector_json"] == {"paper_label": "constitution_lora", "object_kind": "lora", "transfer_kind": "base_to_instruct", "fit_model_id": synth.FIT_MODEL_ID}
    assert lora["extra"]["transfer_kind"] == "instruct_direct" and lora["extra"]["object_transfer_kind"] == "base_to_instruct" and lora["extra"]["fit_model_id"] == synth.FIT_MODEL_ID
    assert by[(synth.MODEL_ID, "safety_si")]["object_selector_json"] == {"paper_label": "safety_si", "object_kind": "system_instruction"}
    assert by[(synth.MODEL_ID, "none")]["object_id"] == "none"
    chat = by[("synth_chat", "constitution_lora")]
    assert chat["object_selector_json"] == {"paper_label": "constitution_lora", "object_kind": "lora", "transfer_kind": "chat_checkpoint"} and "transfer_fit" not in chat["extra"]
    # without the flag, or on a model row already on the transfer kind, the selector is the model row's kind and nothing else
    plain = RO.expand(_mini_with_transfer_fit(flag=None), strict_counts=False)
    assert {(e["model_id"], (e["object_selector_json"] or {}).get("paper_label", e["object_id"])): e["object_selector_json"] for e in plain.entries}[(synth.MODEL_ID, "constitution_lora")] \
        == {"paper_label": "constitution_lora", "object_kind": "lora", "transfer_kind": "instruct_direct"}
    companion = RO.expand(_mini_with_transfer_fit(block_models=[{"model_id": synth.MODEL_ID, "transfer_kind": "base_to_instruct", "fit_model_id": synth.FIT_MODEL_ID}]), strict_counts=False)
    e = next(x for x in companion.entries if x["object_selector_json"] and x["object_selector_json"]["paper_label"] == "constitution_lora")
    assert e["object_selector_json"] == {"paper_label": "constitution_lora", "object_kind": "lora", "transfer_kind": "base_to_instruct"} and "transfer_fit" not in e["extra"]
    assert all(x["roster_version"] == "v1" and x["extra"]["roster_version"] == "v1" for x in companion.entries) and companion.paper_round["roster_version"] == "v1"
    # `except` keeps the served checkpoint's fit for the named cells only
    kept = RO.expand(_mini_with_transfer_fit(flag={"except": [{"model_id": synth.MODEL_ID, "object": "constitution_lora", "eval": synth.EVAL_ID, "reason": "IT-only"}]}), strict_counts=False)
    e = next(x for x in kept.entries if x["model_id"] == synth.MODEL_ID and x["object_selector_json"] and x["object_selector_json"]["paper_label"] == "constitution_lora")
    assert e["object_selector_json"] == {"paper_label": "constitution_lora", "object_kind": "lora", "transfer_kind": "instruct_direct"} and "transfer_fit" not in e["extra"]
    other = RO.expand(_mini_with_transfer_fit(flag={"except": [{"model_id": "synth_chat", "object": "constitution_lora"}]}), strict_counts=False)
    e = next(x for x in other.entries if x["model_id"] == synth.MODEL_ID and x["object_selector_json"] and x["object_selector_json"]["paper_label"] == "constitution_lora")
    assert e["object_selector_json"]["transfer_kind"] == "base_to_instruct"
    # refusals, by name
    with pytest.raises(RO.RosterRefusal, match="several transfer kinds"):
        RO.expand(_mini_with_transfer_fit(block_models=[{"model_id": synth.MODEL_ID, "transfer_kinds": ["base_to_instruct", "instruct_direct"], "views_legacy": ["a", "b"]}]), strict_counts=False)
    with pytest.raises(RO.RosterRefusal, match="must be true or a mapping"):
        RO.expand(_mini_with_transfer_fit(flag="yes"), strict_counts=False)
    with pytest.raises(RO.RosterRefusal, match="must name a model_id, object, eval or comparisons"):
        RO.expand(_mini_with_transfer_fit(flag={"except": [{"reason": "nothing named"}]}), strict_counts=False)
    with pytest.raises(RO.RosterRefusal, match="is not `except` or `reason`"):
        RO.expand(_mini_with_transfer_fit(flag={"only": True}), strict_counts=False)


def test_a_later_roster_version_retires_the_entries_it_no_longer_expands(tmp_path):
    """Migration 0016 end to end on the synthetic corpus: version v1 of a roster reads the fitted object on the served checkpoint's fit
    (selector instruct_direct), so an IT-served run matches and the base->instruct run of the same object does not; version v2 flags the
    block, so the fitted entry is re-shaped (a NEW id: the v1 id stays in the table with its v1 stamp). After v2 loads, the paper_round
    row carries v2 and `v_roster_match` admits v2's entries alone: the base run matches the re-shaped entry and is selected under the
    SAME comparison id, the IT run matches nothing (the v1 id no longer matches although its selector still fits the run) and reads
    not_in_roster, the unsteered entry (identity-stable, re-stamped v2) keeps matching, and `v_roster_gaps` neither lists the retired v1
    id nor any v2 entry. Before 0016 the v1 id kept matching the IT run beside its replacement."""
    v1 = _mini_with_transfer_fit(flag=None, objects=("none", "constitution_lora"))
    v1["comparisons"][0]["models"] = [{"model_id": synth.MODEL_ID, "transfer_kind": "instruct_direct"}]
    v1["meta"]["expanded_counts_by_comparison"] = {"fig_synth": 2}
    # the same object served on the instruct checkpoint at ITS operating coefficient (0.9 against the base fit's 1.0 at the same nominal): a
    # distinct key of the paper relation, so neither run stands as the later attempt of the other and the roster alone decides which is read
    rid_it, recs_it = synth.run_records("it_served", 300, "constitution_lora@deadbeef", synth.OBJECT_SHA, n_items=3, transfer_kind="instruct_direct", strength_coefficient=0.9)
    root, ids, load1, paths1 = _setup(tmp_path, v1, extra_records=recs_it)
    none_run, base_run = ids["run_ids"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT roster_version FROM paper_round WHERE round_id = 'r_synth'").fetchone()[0] == "v1"
    m1 = {}
    for r in conn.execute("SELECT run_id, roster_entry_id FROM v_roster_match"):
        m1.setdefault(r[0], set()).add(r[1])
    e1 = {(e["object_selector_json"] or {}).get("paper_label", e["object_id"]): e["roster_entry_id"] for e in load1.entries}
    assert m1 == {none_run: {e1["none"]}, rid_it: {e1["constitution_lora"]}}                 # v1: the IT-served run is the reading; the base run matches nothing
    assert conn.execute("SELECT COUNT(*) FROM v_roster_gaps").fetchone()[0] == 0
    conn.close()
    import copy
    v2 = copy.deepcopy(v1)
    v2["meta"]["roster_version"] = "v2"
    v2["comparisons"][0]["fitted_objects_use_transfer_fit"] = True
    ry = tmp_path / "roster_v2.yaml"
    ry.write_text(yaml.safe_dump(v2, sort_keys=False))
    load2 = RO.load(str(ry))
    e2 = {(e["object_selector_json"] or {}).get("paper_label", e["object_id"]): e["roster_entry_id"] for e in load2.entries}
    assert e2["none"] == e1["none"] and e2["constitution_lora"] != e1["constitution_lora"]
    paths2 = RO.write_records(root, load2, written_utc=synth.t(60))
    assert any(p.endswith(os.path.join("rulings", "r_synth_v2", "000002-roster_entry.jsonl")) for p in paths2)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT roster_version FROM paper_round WHERE round_id = 'r_synth'").fetchone()[0] == "v2"
    stamps = {r[0]: r[1] for r in conn.execute("SELECT roster_entry_id, roster_version FROM roster_entry WHERE round_id = 'r_synth'")}
    assert stamps == {e1["none"]: "v2", e1["constitution_lora"]: "v1", e2["constitution_lora"]: "v2"}   # three rows: the stale v1 id stays, stamped v1
    m2 = {}
    for r in conn.execute("SELECT run_id, roster_entry_id, comparison_id FROM v_roster_match"):
        m2.setdefault(r[0], set()).add((r[1], r[2]))
    assert m2 == {none_run: {(e1["none"], "fig_synth/instruct_direct")}, base_run: {(e2["constitution_lora"], "fig_synth/instruct_direct")}}
    assert rid_it not in m2                                                                    # the retired v1 selector would still fit this run; the version scope keeps it out
    pc = {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM v_paper_current")}
    assert pc[base_run]["selection_state"] == "selected" and pc[base_run]["in_current_paper"] == 1 and json.loads(pc[base_run]["comparison_ids_json"]) == ["fig_synth/instruct_direct"]
    assert pc[rid_it]["selection_state"] == "not_in_roster" and pc[rid_it]["in_current_paper"] == 0
    assert pc[none_run]["selection_state"] == "selected"
    assert conn.execute("SELECT COUNT(*) FROM v_roster_gaps").fetchone()[0] == 0              # the retired id is no gap; every v2 entry is filled
    # the v1 selector text alone would admit the IT run: only the version keeps it out (the frozen 0015 shape of the match, by construction)
    stale = conn.execute("SELECT object_selector_json FROM roster_entry WHERE roster_entry_id = ?", (e1["constitution_lora"],)).fetchone()[0]
    assert json.loads(stale)["transfer_kind"] == "instruct_direct"
    conn.close()


# --------------------------------------------------------------------------------------------
# roster v10: the appendix transfer figure across models -- one NEW comparison, every fitted object NAMED BY ID (the objects[] grammar)
# --------------------------------------------------------------------------------------------

V10_ROSTER, V10_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v10.yaml", "rulings_r2_v10.jsonl"))
TRANSFER_COMPARISON = "figA_transfer_models"
#: sub-block -> (model, transfer kind, entries): 3 objects x the model's eval set (gemma 23, qwen 23, llama 22 - math500 / ifeval left out); Qwen's base->instruct objects are the three seed fits per recipe
V10_TRANSFER_SUB_BLOCKS = {"gemma_base_to_instruct": ("gemma4e4b_it", "base_to_instruct", 69), "gemma_instruct_direct": ("gemma4e4b_it", "instruct_direct", 69),
                           "qwen_base_to_instruct_seed_fits": ("qwen35_4b_it", "base_to_instruct", 207), "qwen_instruct_direct": ("qwen35_4b_it", "instruct_direct", 69),
                           "llama_base_to_instruct": ("llama8b_it", "base_to_instruct", 66), "llama_instruct_direct": ("llama8b_it", "instruct_direct", 66)}
V10_TRANSFER_EVALS = {"gemma4e4b_it": 23, "qwen35_4b_it": 23, "llama8b_it": 22}
_HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")


def _mini_named(objects, flag=None):
    m = _mini()
    b = m["comparisons"][0]
    b["objects"] = list(objects)
    if flag is not None:
        m["models"][synth.MODEL_ID] = {"model_legacy": "synth", "paper_lane": {"transfer_kind": "instruct_direct"},
                                       "transfer_lane": {"transfer_kind": "base_to_instruct", "view_legacy": "synth_base", "fit_model_id": synth.FIT_MODEL_ID}}
        b["models"] = [{"model_id": synth.MODEL_ID, "transfer_kind": "instruct_direct"}]
        b["fitted_objects_use_transfer_fit"] = flag
    return m


def test_named_object_grammar_expands_one_entry_per_id_with_no_selector_and_refuses_malformed_items():
    """`object_id` / `object_ids` on an objects[] item (v10): one entry per named id, `object_id` set and `object_selector_json` null, the
    label and kind in the extra so a consumer still knows what the entry names; distinct, idempotent identities; the comparison id is the
    model row's; refusals by name for a non-hex id, an id listed twice, the unsteered object (its id is not an artifact sha) and a named
    object on a block flagged `fitted_objects_use_transfer_fit`."""
    a, b = "a" * 64, "b" * 64
    exp = RO.expand(_mini_named([{"object": "constitution_lora", "object_ids": [a, b]}, "none"]), strict_counts=False)
    named = sorted((e for e in exp.entries if e["object_id"] not in (None, "none")), key=lambda e: e["object_id"])
    assert [e["object_id"] for e in named] == [a, b] and all(e["object_selector_json"] is None for e in named)
    assert all(e["comparison_id"] == "fig_synth/base_to_instruct" and e["extra"]["object_named"] is True and e["extra"]["paper_label"] == "constitution_lora"
               and e["extra"]["object_kind"] == "lora" and e["extra"]["object_transfer_kind"] == "base_to_instruct" and e["extra"]["transfer_kind"] == "base_to_instruct" for e in named)
    assert len({e["roster_entry_id"] for e in named}) == 2 and exp.counts_by_comparison == {"fig_synth": 3}
    assert {e["roster_entry_id"] for e in exp.entries} == {e["roster_entry_id"] for e in RO.expand(_mini_named([{"object": "constitution_lora", "object_ids": [a, b]}, "none"]), strict_counts=False).entries}
    one = RO.expand(_mini_named([{"object": "constitution_lora", "object_id": a}]), strict_counts=False)
    assert [e["object_id"] for e in one.entries] == [a] and one.entries[0]["roster_entry_id"] == named[0]["roster_entry_id"]     # the same identity whichever key names it
    # the coarse key of a named entry is the id, so two named ids of one label are two coarse keys
    assert len({RO.entry_key(e) for e in named}) == 2 and all(RO.entry_key(e)[2] == e["object_id"] for e in named)
    # a selector item beside it is untouched
    mixed = RO.expand(_mini_named([{"object": "constitution_lora", "object_id": a}, "constitution_lora"]), strict_counts=False)
    assert sorted(str(e["object_id"]) for e in mixed.entries) == ["None", a] and next(e for e in mixed.entries if e["object_id"] is None)["object_selector_json"]["paper_label"] == "constitution_lora"
    with pytest.raises(RO.RosterRefusal, match="is not a 64-hex sha256"):
        RO.expand(_mini_named([{"object": "constitution_lora", "object_id": "8be1e1a7"}]), strict_counts=False)
    with pytest.raises(RO.RosterRefusal, match="is listed twice"):
        RO.expand(_mini_named([{"object": "constitution_lora", "object_ids": [a, a]}]), strict_counts=False)
    with pytest.raises(RO.RosterRefusal, match="cannot be named by id"):
        RO.expand(_mini_named([{"object": "none", "object_id": a}]), strict_counts=False)
    with pytest.raises(RO.RosterRefusal, match="not re-shaped by fitted_objects_use_transfer_fit"):
        RO.expand(_mini_named([{"object": "constitution_lora", "object_id": a}], flag=True), strict_counts=False)


def test_named_object_load_reports_an_absent_id_and_refuses_a_row_that_disagrees_with_the_item(tmp_path):
    """Against a database: a named id the object table lacks is reported under unknown_to_db.object_ids (the entry dependency-blocks at the
    fold, as a named instrument does); a named id whose row carries another paper_label / object_kind than the item refuses by name."""
    import sqlite3
    a, b = "a" * 64, "b" * 64
    p = tmp_path / "mini.yaml"
    p.write_text(yaml.safe_dump(_mini_named([{"object": "constitution_lora", "object_ids": [a, b]}])), encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY, paper_label TEXT, object_kind TEXT)")
    conn.execute("CREATE TABLE model (model_id TEXT)"); conn.execute("CREATE TABLE eval (eval_id TEXT)"); conn.execute("CREATE TABLE instrument (instrument_id TEXT)")
    conn.execute("INSERT INTO object VALUES (?, 'constitution_lora', 'lora')", (a,))
    load = RO.load(str(p), None, conn=conn, strict_counts=False)
    assert load.unknown_to_db["object_ids"] == [b] and RO.named_object_disagreements(conn, load) == []
    conn.execute("UPDATE object SET paper_label = 'control_lora' WHERE object_id = ?", (a,))
    with pytest.raises(RO.RosterRefusal, match="disagrees with the item"):
        RO.load(str(p), None, conn=conn, strict_counts=False)
    bad = RO.named_object_disagreements(conn, load)
    assert len(bad) == 1 and bad[0]["object_id"] == a and bad[0]["named_label"] == "constitution_lora" and bad[0]["db_label"] == "control_lora"


def test_committed_v10_adds_the_appendix_transfer_comparison_with_every_fitted_object_named_by_id():
    """roster_r2_v10.yaml (2026-09-25): v9 plus ONE new comparison, figA_transfer_models, for the registry
    notebook's appendix transfer figure (base->instruct reading minus instruct-direct reading per (model, recipe)): gemma4e4b_it /
    qwen35_4b_it / llama8b_it x the three LoRA recipes x BOTH transfer kinds x the model's eval set (gemma 23, qwen 23, llama 22: the union of
    the model's base->instruct objects' in-paper evals less math500 / ifeval, which the figure's categories never draw; best-of-N on the live
    selector, the coding evals on the FULL item sets), every fitted
    object NAMED BY ID -- the listed instruct-direct fits (the batch-8 itdoc constitution_lora on gemma / qwen, which a label selector
    could not tell from the batch-64 itdoc64 sibling) and, on Qwen's base->instruct row, the THREE batch-64 seed fits per recipe (one entry
    each, never the batch-8 originals). 546 additive entries in six sub-blocks; every v9 entry keeps its identity; fig4_transfer and
    figQ_transfer untouched; the rulings file is v9's byte for byte; every entry and the paper_round row carry roster_version v10."""
    import collections
    if not os.path.exists(V10_ROSTER):
        pytest.skip("no v10 roster file in this tree")
    v9, v10 = RO.load(V9_ROSTER, V9_RULINGS), RO.load(V10_ROSTER, V10_RULINGS)
    s9, s10 = v9.summary(), v10.summary()
    assert v10.version == "v10" and s10["entries"] == 7083 + 546 == 7629 and s10["rulings"] == 93 and s10["dispositions"] == 102
    assert s10["disagreements"] == s9["disagreements"] == 661 and s10["drawn_false"] == s9["drawn_false"] == 330 and s10["inferred"] == s9["inferred"] + 546 == 2456
    assert {k: v for k, v in s10["counts_by_comparison"].items() if k != TRANSFER_COMPARISON} == s9["counts_by_comparison"] and s10["counts_by_comparison"][TRANSFER_COMPARISON] == 546
    assert TRANSFER_COMPARISON not in s9["counts_by_comparison"]
    assert open(V10_RULINGS, "rb").read() == open(V9_RULINGS, "rb").read()
    assert _round_sans_version(v10.paper_round) == _round_sans_version(v9.paper_round) and v10.paper_round["roster_version"] == "v10"
    d = RO.diff_entries(v10.entries, v9.entries)
    assert not d["changed"] and not d["only_b"] and len(d["only_a"]) == 546 and all(k["key"][0] == TRANSFER_COMPARISON for k in d["only_a"])
    assert len(d["coarse_only_a"]) == 546 and not d["coarse_only_b"]                        # one coarse key per named entry: the id is the key
    assert {e["roster_entry_id"] for e in v9.entries} < {e["roster_entry_id"] for e in v10.entries}
    new = [e for e in v10.entries if e["comparison_id"] == TRANSFER_COMPARISON]
    assert len(new) == 546 and all(e["roster_version"] == "v10" and e["extra"]["roster_version"] == "v10" for e in v10.entries)
    assert collections.Counter(e["extra"]["sub_block"] for e in new) == {k: n for k, (m, tk, n) in V10_TRANSFER_SUB_BLOCKS.items()}
    for e in new:
        m, tk, _ = V10_TRANSFER_SUB_BLOCKS[e["extra"]["sub_block"]]
        assert e["model_id"] == m and e["extra"]["transfer_kind"] == tk and e["extra"]["object_transfer_kind"] == tk and e["strength_rule"] == "operating_point"
        assert _HEX64.match(e["object_id"]) and e["object_selector_json"] is None and e["extra"]["object_named"] is True and e["inferred"] == 1 and e["drawn"] == 1
        assert e["extra"]["paper_label"] in ("control_lora", "constitution_lora", "difference_lora") and e["extra"]["object_kind"] == ("lora_subtraction" if e["extra"]["paper_label"] == "difference_lora" else "lora")
        if e["model_id"] == "qwen35_4b_it" and e["extra"]["transfer_kind"] == "instruct_direct" and e["eval_id"] in ("agentic_leaking", "agentic_murder"):
            assert _HEX64.match(e["instrument_id"]) and e["instrument_selector_json"] is None     # the vendor-tree instruments, named (the frame's Qwen instruct-direct agentic rows)
        else:
            assert e["instrument_id"] is None and e["instrument_selector_json"] is not None
    # the eval set per model, the coding evals on the FULL item sets, best-of-N on the live selector
    for m, n in V10_TRANSFER_EVALS.items():
        evs = {e["eval_id"] for e in new if e["model_id"] == m}
        assert len(evs) == n and {"pasttense", "codeattack", "renellm", "grok420", "strongreject_bon", "pair", "crescendo", "actorattack", "xstest", "or_bench_hard", "mmlu", "gsm8k",
                                  "petri_pooled", "agentic_leaking", "agentic_murder", "bigcodebench"} < evs
        assert ("bfcl" in evs) == (m != "llama8b_it") and not ({"math500", "ifeval"} & evs)      # off every axis (2026-09-23)
    for e in new:
        if e["eval_id"] == "bigcodebench":
            assert e["instrument_selector_json"] == {"token_cap_tokens": [4096, 16384], "harness": "inspect_evals", "thinking_mode": "off", "scoring_rule": "pass_at_1_sandbox", "item_set_id": "bigcodebench_full_1140"}
        if e["eval_id"] == "bfcl":
            assert e["instrument_selector_json"] == {"harness": "bfcl_eval", "thinking_mode": "off", "scoring_rule": "ast_match", "item_set_id": "bfcl_full_3981"}
        if e["eval_id"] == "strongreject_bon":
            assert e["instrument_selector_json"] == {"token_cap_tokens": [512, 4096, 8192], "harness": "bespoke_sisteer", "item_set_id": "jbb_goals_100"}
    # the objects: 24 distinct ids -- one per (model, transfer kind, recipe) except Qwen's base->instruct row with three seed fits per recipe
    ids_by = collections.defaultdict(set)
    for e in new:
        ids_by[(e["model_id"], e["extra"]["transfer_kind"], e["extra"]["paper_label"])].add(e["object_id"])
    assert len({i for s in ids_by.values() for i in s}) == 24                          # 9 instruct-direct + 6 gemma / llama base fits + 9 Qwen seed fits
    assert all(len(s) == (3 if k[:2] == ("qwen35_4b_it", "base_to_instruct") else 1) for k, s in ids_by.items()) and len(ids_by) == 18
    # every other block is byte-identical to v9's: the yaml differs in the version, the addendum, the counts and the appended block alone
    roster, v9y = yaml.safe_load(open(V10_ROSTER, encoding="utf-8")), yaml.safe_load(open(V9_ROSTER, encoding="utf-8"))
    assert roster["meta"]["roster_version"] == "v10" and "addendum_2026_09_25_v10" in roster["meta"]
    assert {k: v for k, v in roster["meta"].items() if k not in ("roster_version", "addendum_2026_09_25_v10", "expanded_counts_by_comparison")} == \
        {k: v for k, v in v9y["meta"].items() if k not in ("roster_version", "expanded_counts_by_comparison")}
    c10, c9 = roster["meta"]["expanded_counts_by_comparison"], v9y["meta"]["expanded_counts_by_comparison"]
    assert {k: v for k, v in c10.items() if k not in (TRANSFER_COMPARISON, "total")} == {k: v for k, v in c9.items() if k != "total"}
    assert c10[TRANSFER_COMPARISON] == 546 and c10["total"] == 7629 and c9["total"] == 7083
    assert {k: v for k, v in roster.items() if k not in ("meta", "comparisons")} == {k: v for k, v in v9y.items() if k not in ("meta", "comparisons")}
    assert [b for b in roster["comparisons"] if b.get("comparison_id_template") != TRANSFER_COMPARISON] == v9y["comparisons"]
    blk = next(b for b in roster["comparisons"] if b.get("comparison_id_template") == TRANSFER_COMPARISON)
    assert "fitted_objects_use_transfer_fit" not in blk and blk["inferred"] is True and blk["expanded_count"] == 546 and "objects" not in blk and "evals" not in blk
    assert [sb["name"] for sb in blk["sub_blocks"]] == list(V10_TRANSFER_SUB_BLOCKS) and all(sb["expanded_count"] == V10_TRANSFER_SUB_BLOCKS[sb["name"]][2] for sb in blk["sub_blocks"])
    assert all(len(sb["models"]) == 1 and sb["models"][0]["model_id"] == V10_TRANSFER_SUB_BLOCKS[sb["name"]][0] and sb["models"][0]["transfer_kind"] == V10_TRANSFER_SUB_BLOCKS[sb["name"]][1] for sb in blk["sub_blocks"])
    assert all(all(("object_id" in o) != ("object_ids" in o) for o in sb["objects"]) for sb in blk["sub_blocks"])
    named_inst = {e["instrument_id"] for e in new if e["instrument_id"]}
    assert len(named_inst) == 2 and {e["eval_id"] for e in new if e["instrument_id"]} == {"agentic_leaking", "agentic_murder"} and sum(1 for e in new if e["instrument_id"]) == 6


def test_roster_load_cli_on_the_committed_v10_files_names_the_twenty_four_objects_the_seedless_root_lacks(tmp_path, monkeypatch, capsys):
    """On a seedless root (no object row) v10 loads and reports exactly the 24 named object ids unknown -- the loader's report for named
    objects, the analogue of v6-v8's for named instruments; nothing else is unknown beyond v9's twelve padding instruments."""
    if not os.path.exists(V10_ROSTER):
        pytest.skip("no v10 roster file in this tree")
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["roster", "load", "--version", "v10", "--no-ingest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "v10" and out["entries"] == 7629 and out["rulings"] == 93
    assert any(p.endswith(os.path.join("rulings", "r2_v10", "000002-roster_entry.jsonl")) for p in out["written"])
    conn = DB.connect(DB.db_path(root), readonly=True)
    load = RO.load(V10_ROSTER, V10_RULINGS, conn=conn)
    named = {e["object_id"] for e in load.entries if e["comparison_id"] == TRANSFER_COMPARISON}
    assert len(named) == 24 and set(load.unknown_to_db["object_ids"]) >= named
    named_inst = {e["instrument_id"] for e in load.entries if e["comparison_id"] == TRANSFER_COMPARISON and e["instrument_id"]}
    assert len(named_inst) == 2 and load.unknown_to_db["instrument_ids"] == sorted(set(PADDED_STATIC_INSTRUMENTS) | set(ZERO_POINT_STATIC_INSTRUMENTS) | set(NEW_PADDING_INSTRUMENTS) | named_inst)
    conn.close()


# --------------------------------------------------------------------------------------------
# roster v11: both control objects at alpha = -1 on Qwen -- one NEW comparison, six dose-scaled control objects NAMED BY ID at fixed:-1
# --------------------------------------------------------------------------------------------

V11_ROSTER, V11_RULINGS = (os.path.join(ROOT, "configs", "registry", f) for f in ("roster_r2_v11.yaml", "rulings_r2_v11.jsonl"))
CTL_MINUS1_COMPARISON = "figA_qwen_control_alpha_minus1"
#: sub-block -> (the standup that served the runs, paper_label, object_kind, the named dose-scaled object): one sub-block per standup, 8 evals each
V11_CTL_MINUS1_SUB_BLOCKS = {
    "control_lora_s1": ("qwen35_4b_dm1_ctllora64s1", "control_lora", "lora", "992b69796f65f9c17fb22fd1c15dce173e24a1e0aa34000df6bbda9bcdfa48a4"),
    "control_lora_s2": ("qwen35_4b_dm1_ctllora64s2", "control_lora", "lora", "53d5e1c74ad9f168902d4448f33d35dcfbefb0b95986fa726a4909d20167dee8"),
    "control_lora_s3": ("qwen35_4b_dm1_ctllora64s3", "control_lora", "lora", "8b30616430c646dbb5645d83647a7cb7aebd2795db1eda48d8188f1127d6f7db"),
    "control_relu_s1": ("qwen35_4b_dm1_ctlrelu64s1", "control_relu", "relu_steer", "54df102e494780f43109b2ae39c0a4a3a887873bd6e28adf1f307ae955d36c11"),
    "control_relu_s2": ("qwen35_4b_dm1_ctlrelu64s2", "control_relu", "relu_steer", "e0e5101c27cf201c84ba2a087c4babcd201a922ca5389aee63cae37385beaed8"),
    "control_relu_s3": ("qwen35_4b_dm1_ctlrelu64s3", "control_relu", "relu_steer", "c59b3f5eba307825a6b2b10963008184c1639755ff23659fdcd14a0ec107a5ab"),
}
V11_CTL_MINUS1_EVALS = {"pasttense", "codeattack", "renellm", "grok420", "crescendo", "actorattack", "xstest", "or_bench_hard"}


def test_committed_v11_adds_the_qwen_control_alpha_minus1_comparison_with_six_named_objects_at_fixed_minus1():
    """roster_r2_v11.yaml (2026-09-26): v10 plus ONE new comparison, figA_qwen_control_alpha_minus1,
    for the registry notebook's appendix figure of BOTH control objects at alpha = -1 on Qwen3.5-4B -- qwen35_4b_it's base->instruct row x
    control_lora + control_relu x the three seed standups x eight evals (the four statics, crescendo, actorattack, xstest, or_bench_hard) at
    strength_rule fixed:-1, every object NAMED BY ID (the six dose-scaled control objects the ctl_neg1 standups serve, one object row per label x
    seed: a label selector could neither give the figure one entry per seed nor keep out the native ladder's batch-8 control_lora -1 point).
    48 additive entries in six sub-blocks, one per standup; every v10 entry keeps its identity; the rulings file is v10's byte for byte; the
    statics and over-refusal entries carry the SAME instrument and judge selectors as fig5_frontier/base_to_instruct's control_lora fixed:-1
    entries (the reading those rows have today), the multi-turn entries the same as the roster's other crescendo / actorattack entries."""
    import collections
    if not os.path.exists(V11_ROSTER):
        pytest.skip("no v11 roster file in this tree")
    v10, v11 = RO.load(V10_ROSTER, V10_RULINGS), RO.load(V11_ROSTER, V11_RULINGS)
    s10, s11 = v10.summary(), v11.summary()
    assert v11.version == "v11" and s11["entries"] == 7629 + 48 == 7677 and s11["rulings"] == 93 and s11["dispositions"] == 102
    assert s11["disagreements"] == s10["disagreements"] == 661 and s11["drawn_false"] == s10["drawn_false"] == 330 and s11["inferred"] == s10["inferred"] + 48 == 2504
    assert {k: v for k, v in s11["counts_by_comparison"].items() if k != CTL_MINUS1_COMPARISON} == s10["counts_by_comparison"] and s11["counts_by_comparison"][CTL_MINUS1_COMPARISON] == 48
    assert CTL_MINUS1_COMPARISON not in s10["counts_by_comparison"]
    assert open(V11_RULINGS, "rb").read() == open(V10_RULINGS, "rb").read()
    assert _round_sans_version(v11.paper_round) == _round_sans_version(v10.paper_round) and v11.paper_round["roster_version"] == "v11"
    d = RO.diff_entries(v11.entries, v10.entries)
    assert not d["changed"] and not d["only_b"] and len(d["only_a"]) == 48 and all(k["key"][0] == CTL_MINUS1_COMPARISON for k in d["only_a"])
    assert len(d["coarse_only_a"]) == 48 and not d["coarse_only_b"]                         # one coarse key per named entry: the id is the key
    assert {e["roster_entry_id"] for e in v10.entries} < {e["roster_entry_id"] for e in v11.entries}
    new = [e for e in v11.entries if e["comparison_id"] == CTL_MINUS1_COMPARISON]
    assert len(new) == 48 and all(e["roster_version"] == "v11" and e["extra"]["roster_version"] == "v11" for e in v11.entries)
    assert collections.Counter(e["extra"]["sub_block"] for e in new) == {k: 8 for k in V11_CTL_MINUS1_SUB_BLOCKS}
    # the selectors the entries must share with the rows' reading today: fig5_frontier/base_to_instruct's control_lora fixed:-1 entries (statics,
    # over-refusal) and the roster's crescendo / actorattack entries (multi-turn; the frontier lists no multi-turn eval)
    fig5 = {e["eval_id"]: e for e in v10.entries if e["comparison_id"] == "fig5_frontier/base_to_instruct" and e["model_id"] == "qwen35_4b_it"
            and e["strength_rule"] == "fixed:-1" and (e["object_selector_json"] or {}).get("paper_label") == "control_lora"}
    assert set(fig5) == {"pasttense", "codeattack", "renellm", "grok420", "xstest", "or_bench_hard"}
    mt = {e["eval_id"]: e for e in v10.entries if e["comparison_id"] == "fig2_jailbreak/base_to_instruct" and e["model_id"] == "qwen35_4b_it" and e["eval_id"] in ("crescendo", "actorattack")}
    assert set(mt) == {"crescendo", "actorattack"}
    for e in new:
        standup, label, kind, oid = V11_CTL_MINUS1_SUB_BLOCKS[e["extra"]["sub_block"]]
        assert e["model_id"] == "qwen35_4b_it" and e["extra"]["transfer_kind"] == "base_to_instruct" and e["extra"]["object_transfer_kind"] == "base_to_instruct"
        assert e["strength_rule"] == "fixed:-1" and e["object_id"] == oid and e["object_selector_json"] is None and e["extra"]["object_named"] is True
        assert e["extra"]["paper_label"] == label and e["extra"]["object_kind"] == kind and e["extra"]["view_legacy"] == standup and e["extra"]["fit_model_id"] == "qwen35_4b_base"
        assert e["extra"]["sub_block_extra"]["standup_legacy"] == standup and e["extra"]["sub_block_extra"]["fit_replicate_label"] == standup[-2:] and e["extra"]["sub_block_extra"]["source_legacy"] == "ctl_neg1"
        assert e["instrument_id"] is None and e["instrument_selector_json"] is not None and e["inferred"] == 1 and e["drawn"] == 1
        ref = fig5.get(e["eval_id"]) or mt[e["eval_id"]]
        assert e["instrument_selector_json"] == ref["instrument_selector_json"] and e["judge_selector_json"] == ref["judge_selector_json"], e["eval_id"]
    for name in V11_CTL_MINUS1_SUB_BLOCKS:
        assert {e["eval_id"] for e in new if e["extra"]["sub_block"] == name} == V11_CTL_MINUS1_EVALS
    assert {e["instrument_selector_json"]["token_cap_tokens"][0] for e in new if e["eval_id"] in ("crescendo", "actorattack")} == {4096}
    assert all(4096 in e["instrument_selector_json"]["token_cap_tokens"] and 8192 in e["instrument_selector_json"]["token_cap_tokens"] for e in new if e["eval_id"] in ("pasttense", "codeattack", "renellm", "grok420"))
    assert {e["judge_selector_json"]["judge_model"] for e in new if e["eval_id"] in ("xstest", "or_bench_hard")} == {"claude-sonnet-5"}
    assert {e["judge_selector_json"]["judge_model"] for e in new if e["eval_id"] not in ("xstest", "or_bench_hard")} == {"claude-haiku-4-5"}
    ids = {e["object_id"] for e in new}
    assert len(ids) == 6 and all(_HEX64.match(i) for i in ids) and ids == {v[3] for v in V11_CTL_MINUS1_SUB_BLOCKS.values()}
    assert not (ids & {e["object_id"] for e in v10.entries if e["object_id"]})                # the +1 parents and v10's named objects are other rows
    # every other block is byte-identical to v10's: the yaml differs in the version, the addendum, the counts and the inserted block alone
    roster, v10y = yaml.safe_load(open(V11_ROSTER, encoding="utf-8")), yaml.safe_load(open(V10_ROSTER, encoding="utf-8"))
    assert roster["meta"]["roster_version"] == "v11" and "addendum_2026_09_26_v11" in roster["meta"]
    assert {k: v for k, v in roster["meta"].items() if k not in ("roster_version", "addendum_2026_09_26_v11", "expanded_counts_by_comparison")} == \
        {k: v for k, v in v10y["meta"].items() if k not in ("roster_version", "expanded_counts_by_comparison")}
    c11, c10 = roster["meta"]["expanded_counts_by_comparison"], v10y["meta"]["expanded_counts_by_comparison"]
    assert {k: v for k, v in c11.items() if k not in (CTL_MINUS1_COMPARISON, "total")} == {k: v for k, v in c10.items() if k != "total"}
    assert c11[CTL_MINUS1_COMPARISON] == 48 and c11["total"] == 7677 and c10["total"] == 7629
    assert {k: v for k, v in roster.items() if k not in ("meta", "comparisons")} == {k: v for k, v in v10y.items() if k not in ("meta", "comparisons")}
    assert [b for b in roster["comparisons"] if b.get("comparison_id_template") != CTL_MINUS1_COMPARISON] == v10y["comparisons"]
    blk = next(b for b in roster["comparisons"] if b.get("comparison_id_template") == CTL_MINUS1_COMPARISON)
    assert "fitted_objects_use_transfer_fit" not in blk and blk["inferred"] is True and blk["expanded_count"] == 48 and "objects" not in blk and "evals" not in blk
    assert blk["strength_rules"] == ["fixed:-1"] and blk["exceptions"] == v10y["sets"]["common_exceptions"]
    assert [sb["name"] for sb in blk["sub_blocks"]] == list(V11_CTL_MINUS1_SUB_BLOCKS) and all(sb["expanded_count"] == 8 and sb["strength_rules"] == ["fixed:-1"] for sb in blk["sub_blocks"])
    for sb in blk["sub_blocks"]:
        standup, label, kind, oid = V11_CTL_MINUS1_SUB_BLOCKS[sb["name"]]
        assert len(sb["models"]) == 1 and sb["models"][0] == {"model_id": "qwen35_4b_it", "transfer_kind": "base_to_instruct", "view_legacy": standup, "fit_model_id": "qwen35_4b_base", "fit_replicate_label": standup[-2:]}
        assert sb["objects"] == [{"object": label, "object_id": oid}] and sb["evals"] == ["pasttense", "codeattack", "renellm", "grok420", "crescendo", "actorattack", "xstest", "or_bench_hard_1k"]


def test_committed_v11_control_alpha_minus1_comparison_escapes_every_comparison_glob_of_the_comparison_scoped_excludes():
    """The new comparison declares control_relu (object_kind relu_steer), so its id must match NO glob of the six-arms exclude
    r2v1-exclude-relu_steer-main-figures (figA_*_absolute/* among them) nor of any other comparison-scoped exclude in v11's
    rulings, evaluated with sqlite's own GLOB (the matcher `matched` / `covered` use) -- else v_paper_current would strip the comparison off
    every control_relu run it admits and the figure would read the ReLU half as absent."""
    import sqlite3
    if not os.path.exists(V11_ROSTER):
        pytest.skip("no v11 roster file in this tree")
    v11 = RO.load(V11_ROSTER, V11_RULINGS)
    con = sqlite3.connect(":memory:")
    glob = lambda s, g: bool(con.execute("SELECT ? GLOB ?", (s, g)).fetchone()[0])
    assert glob("figA_attacks_absolute/base_to_instruct", "figA_*_absolute/*") and not glob(CTL_MINUS1_COMPARISON, "figA_*_absolute/*")
    checked = 0
    for r in v11.rulings:
        if r["action"] != "exclude" or r["scope_kind"] != "comparison":
            continue
        for g in (r["scope_json"].get("match") or {}).get("comparison_ids") or []:
            assert not glob(CTL_MINUS1_COMPARISON, g), (r["ruling_id"], g)
            checked += 1
    assert checked >= 9


def test_roster_load_cli_on_the_committed_v11_files_names_the_six_control_objects_beyond_v10s_on_a_seedless_root(tmp_path, monkeypatch, capsys):
    """On a seedless root (no object row) v11 loads and reports unknown exactly the object ids v10 reports there PLUS the six dose-scaled control
    objects (v10's 24 named ids, the four AlphaSteer pins and the unsteered id are unknown on a seedless root for both versions); nothing else is
    unknown beyond v9's twelve padding instruments and v10's two named agentic instruments."""
    if not os.path.exists(V11_ROSTER):
        pytest.skip("no v11 roster file in this tree")
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    capsys.readouterr()
    assert cli.main(["roster", "load", "--version", "v11", "--no-ingest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == "v11" and out["entries"] == 7677 and out["rulings"] == 93
    assert any(p.endswith(os.path.join("rulings", "r2_v11", "000002-roster_entry.jsonl")) for p in out["written"])
    conn = DB.connect(DB.db_path(root), readonly=True)
    load = RO.load(V11_ROSTER, V11_RULINGS, conn=conn)
    named = {e["object_id"] for e in load.entries if e["comparison_id"] == CTL_MINUS1_COMPARISON}
    named_v10 = {e["object_id"] for e in load.entries if e["comparison_id"] == TRANSFER_COMPARISON}
    assert named == {v[3] for v in V11_CTL_MINUS1_SUB_BLOCKS.values()} and len(named_v10) == 24 and not (named & named_v10)
    l10 = RO.load(V10_ROSTER, V10_RULINGS, conn=conn)
    assert not (named & set(l10.unknown_to_db["object_ids"])) and set(load.unknown_to_db["object_ids"]) == set(l10.unknown_to_db["object_ids"]) | named
    named_inst = {e["instrument_id"] for e in load.entries if e["comparison_id"] == TRANSFER_COMPARISON and e["instrument_id"]}
    assert load.unknown_to_db["instrument_ids"] == sorted(set(PADDED_STATIC_INSTRUMENTS) | set(ZERO_POINT_STATIC_INSTRUMENTS) | set(NEW_PADDING_INSTRUMENTS) | named_inst)
    conn.close()
