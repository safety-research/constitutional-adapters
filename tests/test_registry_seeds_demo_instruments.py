"""The front-door WORKED EXAMPLE's decode rows and instrument bindings (ruling `decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09`).

The public reproduction recipe submits the worked example's runs through the registry planner (`jobplan.canonical_instrument`), which
binds a model-specific instrument through `instruments.yaml decode_policy_refs` -> the instrument record's `extra.model_ids`. The key
`qwen35_4b_demo` (tests/test_registry_seeds_demo_model.py) had no decode row and no ref, so the planner minted nothing for
it. The front-door session's ruling of 2026-10-09: the worked example declares no eval and no preset of its own -- it is the paper's
Qwen3.5-4B checkpoint under the same pins, running the paper's instruments under the paper's presets -- so its rows are `qwen35_4b_it`'s,
with the key swapped. This module pins the mirror from both sides:

* `configs/registry/decode_policy_seed.yaml`: every explicit `qwen35_4b_it|...` row has exactly one `qwen35_4b_demo|...` twin and no
  twin lacks its original (a future 4B change that forgets the twin fails BY NAME); every compared field is equal, `same_values_as` points
  at the 4B's row (the loader checks the numbers), the twin is bound to the ruling, supersession is mirrored twin-to-twin, and the
  default flags are the 4B's; the five `(qwen35_4b, mode)` sampling-mode entries have `(qwen35_4b_demo, mode)` twins;
* `configs/registry/instruments.yaml`: every `decode_policy_refs` list naming a 4B ref names the twin right after it;
* the binding reaches the planner: a twin states the 4B's sent numbers, so its instrument record hashes to the 4B's identity and the
  loader merges them into ONE record whose `extra.model_ids` lists both keys (`dimensions.build_records`, the rule gemma4_31b's rows
  follow into gemma4e4b_it's); `canonical_instrument(idx, eval, "qwen35_4b_demo", path)` therefore returns the SAME record as for the
  4B wherever the 4B is bound through an explicit ref. The `*` wildcard rows expand over served models only (`served_model_ids`:
  in_paper / descoped), so a `reference` key gets nothing from them -- the greedy controls, the MMLU letter read and the protocol rows
  have no twin and keep refusing by name; what the 4B resolves and the demo does not is at most that wildcard-bound set;
* the committed export resolves the demo key to the twin's numbers for every mode and family the 4B resolves through its own rows.
"""
from __future__ import annotations

import os
from typing import Dict, Set, Tuple

import pytest
import yaml

from sisteer.registry import decode_policy as DP
from sisteer.registry import jobplan as JP
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.adapters import dimensions as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
EXPORT = os.path.join(ROOT, "configs", "registry", "decode_policy.json")
PROD = synth.producer("tests.registry_seeds_demo_instruments")

DEMO, PAPER = "qwen35_4b_demo", "qwen35_4b_it"
DEMO_KEY, PAPER_KEY = "qwen35_4b_demo", "qwen35_4b"            # the sampling_mode_policy model_key spellings
RULING = "decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09"
#: the fields a twin may spell differently from its 4B row: the key itself, the binding to this ruling, the provenance prose pointing
#: back at the 4B's row, and the two references that are re-keyed (checked explicitly below); every other field is compared verbatim
TWIN_MAY_DIFFER = {"policy_ref", "model_id", "ruling_id", "source_citation", "notes", "seed_rule_detail", "same_values_as",
                   "superseded_by_policy_id"}
#: the ONE demo row with no explicit 4B original: the recipe runs MMLU for every key (reproduce/lib/stage.py FAMILIES) and registers a front-door
#: key natively under the full 1,531-item instrument, and the 4B's MMLU row is a `*` wildcard expansion no `reference` key receives -- so the
#: demo carries an explicit greedy row, same_values_as the served models' expansion (the a3b / 31b shape), named by that instrument ONLY
MMLU_TWIN = f"{DEMO}|mmlu|off|greedy"
MMLU_SIBLING = f"{PAPER}|mmlu|off|greedy"               # the 4B's wildcard expansion: the numbers the twin must equal
MMLU_SAME_VALUES_AS = "gemma4e4b_it|mmlu|off|greedy"
MMLU_INSTRUMENT = "mmlu_letter_prefill_full_1531"
#: the three cap-4,096 multi-turn evals: the paper's seeds (multiturn_*_cap4096_generation_config) name the 4B's AS-FILED
#: `|multi_turn|off|server_default` row (sends nothing, unseeded: decode_seed_rule none); the demo binds its `|multi_turn|off|vendor_preset` twin
#: -- the six-key vendor row a fresh run sends (target_decode.registry_vendor_row refuses a vendor row that does not send all six), seeded per
#: (goal, turn) -- through the vendor-preset seeds (multiturn_*_cap4096_vendor_preset, decode_seed_rule per_goal_turn; the registry owner's
#: split by decode route, 2026-10-14; before it the same refs sat in the paper's seeds under `none`), so the demo's three records are its own,
#: merging with nobody's (the front door's owner, 2026-10-09)
MT_VENDOR_EVALS = ("crescendo", "pair", "actorattack")
MT_PAPER_REF, MT_DEMO_REF = f"{PAPER}|multi_turn|off|server_default", f"{DEMO}|multi_turn|off|vendor_preset"


def _swap(ref: str) -> str:
    assert ref.startswith(PAPER + "|"), ref
    return DEMO + ref[len(PAPER):]


def _unswap(ref: str) -> str:
    assert ref.startswith(DEMO + "|"), ref
    return PAPER + ref[len(DEMO):]


@pytest.fixture(scope="module")
def seeds():
    return D.load_seeds(SEEDS)


@pytest.fixture(scope="module")
def bundle(seeds):
    return D.build_records(seeds, PROD, now="2026-10-09T00:00:00.000Z")


@pytest.fixture(scope="module")
def seed_index():
    return JP.load_seed_index(SEEDS)


@pytest.fixture(scope="module")
def pf():
    return DP.PolicyFile.load(EXPORT)


@pytest.fixture(scope="module")
def paper_rows(seeds) -> Dict[str, dict]:
    return {str(r["policy_ref"]): r for r in seeds.decode["policies"] if str(r.get("policy_ref", "")).startswith(PAPER + "|")}


@pytest.fixture(scope="module")
def demo_rows(seeds) -> Dict[str, dict]:
    return {str(r["policy_ref"]): r for r in seeds.decode["policies"] if str(r.get("policy_ref", "")).startswith(DEMO + "|")}


# --------------------------------------------------------------------------------------------
# the ruling, and the seed rows: one twin per 4B row, identical in every compared field
# --------------------------------------------------------------------------------------------


def test_the_ruling_is_seeded_with_its_date_and_names_the_mirror(seeds):
    text = seeds.decode["rulings"][RULING]
    assert "2026-10-09" in text and "worked example" in text and PAPER in text and DEMO in text
    assert "same_values_as" in text and "canonical_instrument" in text


def test_every_4b_row_has_exactly_one_demo_twin_and_no_twin_lacks_its_original(paper_rows, demo_rows):
    assert paper_rows, "the 4B rows are the template"
    assert set(demo_rows) == {_swap(r) for r in paper_rows} | {MMLU_TWIN}, {
        "4B rows without a twin": sorted(set(paper_rows) - {_unswap(r) for r in demo_rows}),
        "twins without a 4B row": sorted(set(demo_rows) - {_swap(r) for r in paper_rows} - {MMLU_TWIN})}
    assert MMLU_SIBLING not in paper_rows                     # the 4B's MMLU row is a wildcard expansion, not an explicit row
    mm = demo_rows[MMLU_TWIN]
    assert mm["same_values_as"] == MMLU_SAME_VALUES_AS and mm["ruling_id"] == RULING and mm["mode"] == mm["decode_source"] == "greedy"
    assert mm["is_default_for_lookup"] == 1 and mm["eval_id"] == "mmlu" and "superseded_by_policy_id" not in mm


def test_each_twin_is_its_4b_row_with_the_key_swapped(paper_rows, demo_rows):
    for ref, prow in sorted(paper_rows.items()):
        trow = demo_rows[_swap(ref)]
        for k in sorted((set(prow) | set(trow)) - TWIN_MAY_DIFFER):
            assert trow.get(k) == prow.get(k), (ref, k, prow.get(k), trow.get(k))
        assert trow["model_id"] == DEMO and trow["ruling_id"] == RULING, ref
        assert trow["same_values_as"] == ref, (ref, trow.get("same_values_as"))
        assert str(trow["source_citation"]).startswith(f"as {ref}") and RULING in str(trow["source_citation"]), ref
        if "notes" in trow:                                   # a twin of a row without notes stays without (verbatim, key swapped)
            assert ref in str(trow["notes"]), ref
        if prow.get("superseded_by_policy_id"):
            assert trow["superseded_by_policy_id"] == _swap(str(prow["superseded_by_policy_id"])), ref
        else:
            assert not trow.get("superseded_by_policy_id"), ref
        assert ("seed_rule_detail" in trow) == ("seed_rule_detail" in prow), ref


def test_the_loader_accepts_the_twins_and_mirrors_default_supersession_and_numbers(bundle, paper_rows):
    idx = bundle.policies
    twins = {pid: pe for pid, pe in idx.rows.items() if pid.startswith(DEMO + "|")}
    assert len(twins) == len(paper_rows) + 1                   # + the explicit MMLU greedy row
    for pid, (p, e) in sorted(twins.items()):
        sib = _unswap(pid)
        sp, se = idx.rows[sib]
        if pid == MMLU_TWIN + "@v1":
            assert e["same_values_as"] == MMLU_SAME_VALUES_AS + "@v1" and se.get("wildcard_expanded_from") == "*|mmlu|off|greedy", pid
        else:
            assert e["same_values_as"] == sib, pid
        assert idx.numbers(pid) == idx.numbers(sib), pid
        assert e["decode_ruling_ref"] == RULING and "2026-10-09" in e["decode_ruling_text"], pid
        assert int(p.get("is_default") or 0) == int(sp.get("is_default") or 0), pid
        assert not e.get("wildcard_expanded_from") and not e.get("superseded_by_unresolved"), pid
        if sp.get("superseded_by_policy_id"):
            assert p["superseded_by_policy_id"] == _swap(str(sp["superseded_by_policy_id"])), pid
        else:
            assert not p.get("superseded_by_policy_id"), pid
        for k in ("decode_source", "thinking_mode", "eval_id", "eval_category", "seed_rule", "effective_from_utc", "max_new_tokens_rule"):
            if pid == MMLU_TWIN + "@v1" and k == "effective_from_utc":
                continue                                       # the explicit row carries its ruling's date; the wildcard expansion is `unverified`
            assert p.get(k) == sp.get(k), (pid, k)
        assert e.get("params_sent") == se.get("params_sent") and e.get("mode") == se.get("mode"), pid
    assert not [k for k in idx.no_default_keys if k.startswith(DEMO + "|")], idx.no_default_keys
    assert not [w for w in bundle.warnings if "demo" in w], bundle.warnings


def test_the_sampling_mode_entries_of_the_4b_key_have_demo_twins(seeds, bundle):
    ents = [e for e in seeds.decode["sampling_mode_policy"] if e.get("model_key") in (PAPER_KEY, DEMO_KEY)]
    paper = {e["mode"]: e for e in ents if e["model_key"] == PAPER_KEY}
    demo = {e["mode"]: e for e in ents if e["model_key"] == DEMO_KEY}
    assert set(paper) == set(demo) and paper, (sorted(paper), sorted(demo))
    for mode, pe in paper.items():
        assert demo[mode]["policy_ref"] == _swap(str(pe["policy_ref"])) and not demo[mode].get("refuses"), mode
    recs = {(r["payload"]["model_id"], r["payload"]["sampling_mode"]): r["payload"]["policy_id"] for r in bundle.records["sampling_mode_policy"]}
    for mode in paper:
        assert recs[(DEMO, mode)] == _swap(recs[(PAPER, mode)]), mode


# --------------------------------------------------------------------------------------------
# instruments.yaml: the twin ref beside every 4B ref, and the binding reaching the planner
# --------------------------------------------------------------------------------------------


def _instrument_refs() -> Dict[str, list]:
    with open(os.path.join(SEEDS, "instruments.yaml"), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {str(row["instrument_id"]): list(row.get("decode_policy_refs") or []) for row in data["instruments"]}


def test_every_decode_policy_refs_list_naming_a_4b_ref_names_the_twin_right_after_it():
    n_paper = n_demo = 0
    for iid, refs in sorted(_instrument_refs().items()):
        for i, ref in enumerate(refs):
            if ref == MT_PAPER_REF and iid.startswith("multiturn_") and iid.endswith("_cap4096_generation_config"):
                # the 4B's as-filed row stays in the paper's seed; the demo's twin is NOT beside it -- since the 2026-10-14 split by decode
                # route the demo binds the vendor row (the row a fresh run sends, seeded per (goal, turn)) through the matching
                # multiturn_*_cap4096_vendor_preset seed, whose first ref it is (no 4B ref before it: the 4B's vendor row is bound by no seed)
                n_paper += 1
                assert MT_DEMO_REF not in refs and refs[i + 1].startswith("nemotron3nano4b|"), (iid, refs)
                assert MT_DEMO_REF in _instrument_refs()[iid.replace("_generation_config", "_vendor_preset")], iid
            elif ref == MT_DEMO_REF:
                n_demo += 1
                assert iid.startswith("multiturn_") and iid.endswith("_cap4096_vendor_preset") and i == 0, (iid, refs)
                assert MT_PAPER_REF not in refs and not any(r.startswith(PAPER + "|") for r in refs), (iid, refs)
            elif ref.startswith(PAPER + "|"):
                n_paper += 1
                assert i + 1 < len(refs) and refs[i + 1] == _swap(ref), (iid, ref, refs)
            elif ref == MMLU_TWIN:
                assert iid == MMLU_INSTRUMENT, (iid, refs)        # named by the full 1,531-item instrument ONLY (the key stays unambiguous)
            elif ref.startswith(DEMO + "|"):
                n_demo += 1
                assert i > 0 and refs[i - 1] == _unswap(ref), (iid, ref, refs)
    assert n_paper == n_demo > 0
    assert MMLU_TWIN in _instrument_refs()[MMLU_INSTRUMENT] and MMLU_TWIN not in _instrument_refs()["mmlu_letter_prefill_200"]


def test_a_twins_instrument_record_is_the_4b_rows_record_with_both_keys_in_model_ids(bundle):
    recs = [r["payload"] for r in bundle.records["instrument"]]
    with_demo = [p for p in recs if DEMO in (p["extra"].get("model_ids") or [])]
    assert with_demo
    own = [p for p in with_demo if p["eval_id"] in MT_VENDOR_EVALS]
    assert sorted(p["eval_id"] for p in own) == sorted(MT_VENDOR_EVALS)
    for p in own:                                                   # the three multi-turn records are the demo's own
        assert p["extra"]["model_ids"] == [DEMO] and p["extra"]["model_id"] == DEMO and p["extra"]["decode_policy_ref"] == MT_DEMO_REF, p["eval_id"]
    for p in with_demo:
        if p["eval_id"] in MT_VENDOR_EVALS:
            continue
        ids = p["extra"]["model_ids"]
        assert PAPER in ids and DEMO in ids and "model_id" not in p["extra"], (p["extra"].get("seed_instrument_id"), ids)
        pids = p["extra"].get("decode_policy_ids") or []
        assert any(x.startswith(DEMO + "|") for x in pids) and any(x.startswith(PAPER + "|") for x in pids), pids
    # every seed instrument row that names a 4B ref has at least one record carrying the demo key -- except the three cap-4096 multi-turn
    # seeds, whose demo binding moved to the vendor-preset seeds with the 2026-10-14 split by decode route (the 4B's as-filed server_default
    # row sends nothing, unseeded; the demo's vendor twin sends six keys, seeded per (goal, turn): two seed rules, two seeds)
    seeds_with_4b = {iid for iid, refs in _instrument_refs().items() if any(r.startswith(PAPER + "|") for r in refs)}
    carrying = {p["extra"].get("seed_instrument_id") for p in with_demo}
    mt_paper_seeds = {f"multiturn_{e}_cap4096_generation_config" for e in MT_VENDOR_EVALS}
    assert mt_paper_seeds <= seeds_with_4b and not (mt_paper_seeds & carrying), sorted(mt_paper_seeds & carrying)
    assert {f"multiturn_{e}_cap4096_vendor_preset" for e in MT_VENDOR_EVALS} <= carrying, sorted(carrying)
    assert seeds_with_4b - mt_paper_seeds <= carrying, sorted(seeds_with_4b - mt_paper_seeds - carrying)


def _resolved(idx: JP.SeedIndex, model_id: str) -> Dict[Tuple[str, str], Tuple[str, dict, dict]]:
    out = {}
    for eval_id in sorted(idx.evals):
        for path in V.SERVING_PATHS:
            try:
                payload, extra = JP.canonical_instrument(idx, eval_id, model_id, path)
            except LookupError:
                continue
            out[(eval_id, path)] = (str(payload["instrument_id"]), payload, extra)
    return out


def test_the_planner_mints_the_same_canonical_instrument_for_the_demo_wherever_the_4b_is_bound_explicitly(seed_index, bundle):
    policies = bundle.policies.rows
    paper = _resolved(seed_index, PAPER)
    demo = _resolved(seed_index, DEMO)
    assert paper, "the 4B resolves somewhere"
    explicit: Set[Tuple[str, str]] = set()
    wildcard_only: Set[Tuple[str, str]] = set()
    for key, (iid, payload, extra) in paper.items():
        pids = extra.get("decode_policy_ids") or ([extra["decode_policy_id"]] if extra.get("decode_policy_id") else [])
        mine = [pid for pid in pids if policies[pid][0]["model_id"] == PAPER]
        assert mine, (key, pids)
        if any(not policies[pid][1].get("wildcard_expanded_from") for pid in mine):
            explicit.add(key)
        else:
            wildcard_only.add(key)
    assert explicit, "the 4B is bound through its own rows somewhere"
    for key in sorted(explicit):
        assert key in demo, f"the planner mints no canonical instrument for eval {key[0]!r} on {DEMO} via {key[1]}"
        if key[0] in MT_VENDOR_EVALS:
            # the demo's own record: since the 2026-10-14 split by decode route the vendor-preset seed (per_goal_turn), a different identity
            # (the vendor twin's sent numbers AND the seed rule), the demo alone; the paper's 4B record stays in the generation_config seed
            assert demo[key][0] != paper[key][0], key
            assert paper[key][2]["seed_instrument_id"] == f"multiturn_{key[0]}_cap4096_generation_config" and paper[key][1]["decode_seed_rule"] == "none", key
            assert demo[key][2]["seed_instrument_id"] == f"multiturn_{key[0]}_cap4096_vendor_preset" and demo[key][1]["decode_seed_rule"] == "per_goal_turn", key
            assert demo[key][2]["model_ids"] == [DEMO] and demo[key][2]["decode_policy_ref"] == MT_DEMO_REF, key
            continue
        assert demo[key][0] == paper[key][0], key
        ids = demo[key][2].get("model_ids") or []
        assert PAPER in ids and DEMO in ids, (key, ids)
    # beyond the 4B's bindings the demo resolves exactly MMLU (its explicit greedy row, the full 1,531-item instrument -- where the paper's served
    # models are deliberately ambiguous between the 200 and the full instrument); what the 4B resolves but the demo does not is at most the
    # wildcard-bound set
    only_demo = set(demo) - set(paper)
    assert only_demo and {e for e, _ in only_demo} == {"mmlu"}, sorted(only_demo)
    for key in sorted(only_demo):
        assert demo[key][2].get("seed_instrument_id") == MMLU_INSTRUMENT and DEMO in demo[key][2]["model_ids"], key
        assert key not in paper and all(k[0] != "mmlu" for k in paper), "the paper's served models resolve no single MMLU instrument"
    assert set(paper) - set(demo) <= wildcard_only, sorted((set(paper) - set(demo)) - wildcard_only)
    # the families the ruling names are among the explicit bindings
    names = {e for e, _ in explicit}
    assert {"pasttense", "codeattack", "renellm", "grok420", "strongreject_bon", "crescendo", "pair", "actorattack", "xstest", "or_bench_hard",
            "petri_pooled", "gsm8k"} <= names, sorted(names)


# --------------------------------------------------------------------------------------------
# the committed export: the demo key resolves to the twin's numbers
# --------------------------------------------------------------------------------------------


RESOLVE_KWS = (dict(eval_category="static_attack", thinking_mode="off"), dict(eval_category="over_refusal", thinking_mode="off"),
               dict(eval_category="multi_turn", thinking_mode="off"), dict(eval_category="agentic", thinking_mode="off"),
               dict(eval_category="petri", thinking_mode="off"), dict(eval_id="gsm8k", thinking_mode="on"), dict(eval_id="math500", thinking_mode="on"),
               dict(eval_id="ifeval", thinking_mode="off"), dict(eval_id="em8q", thinking_mode="off"), dict(eval_id="strongreject_bon"),
               dict(eval_id="mmlu", thinking_mode="off"),
               dict(mode="card"), dict(mode="thinking_card"), dict(mode="general"), dict(mode="battery"), dict(mode="vendor_preset"))


def test_the_export_resolves_the_demo_key_to_the_twins_numbers(pf):
    assert pf.models[DEMO]["paper_status"] == "reference"
    for kw in RESOLVE_KWS:
        theirs = pf.resolve(PAPER, **kw)
        mine = pf.resolve(DEMO, **kw)
        assert mine.policy_id == _swap(theirs.policy_id) and mine.model_id == DEMO, (kw, mine.policy_id, theirs.policy_id)
        assert mine.params == theirs.params and mine.params_not_sent == theirs.params_not_sent and mine.decode_source == theirs.decode_source, kw
    # a superseded 4B row's twin is readable by name and superseded by the successor's twin
    sup = next(p for p in pf.policies if p["policy_id"] == f"{DEMO}|multi_turn|off|server_default@v1")
    assert sup["superseded_by_policy_id"] == f"{DEMO}|multi_turn|off|vendor_preset@v1" and int(sup["is_default"]) == 0
    # the wildcard rows do not reach a `reference` key (the shape's premise, tests/test_registry_seeds_test_models.py): the paper's key
    # resolves MMLU / BigCodeBench / BFCL through the expanded `*|…` eval rows. MMLU, which the recipe runs, is the demo's explicit greedy row
    # (checked in RESOLVE_KWS above). BigCodeBench / BFCL are not in the recipe's families and have no demo row; pinned AS IT STANDS: the
    # export's eval-over-category rule (C18) then falls to the category default, the demo's `capability|off|card` twin (0.7 / 0.95 / 20) --
    # NOT the protocol numbers, and not a refusal. The planner is the gate there: it mints no canonical instrument for those two on the demo.
    assert pf.resolve(DEMO, eval_id="mmlu", thinking_mode="off").policy_id == MMLU_TWIN + "@v1"
    for eval_id, tail in (("bigcodebench", "|bigcodebench|off|protocol@v1"), ("bfcl", "|bfcl|off|protocol@v1")):
        assert pf.resolve(PAPER, eval_id=eval_id, thinking_mode="off").policy_id == PAPER + tail
        assert pf.resolve(DEMO, eval_id=eval_id, thinking_mode="off").policy_id == f"{DEMO}|capability|off|card@v1", eval_id
