"""The worked example's battery, as the freeze reads it: the EXACT registering eval list and the unit count per family, and a
canonical instrument for every one of those evals minted from the seeds alone.

`reproduce/lib/stage.py expected_units` (the `freeze-inventory` verb `05 --step freeze` runs) freezes `unit_plan(<key>, <baseline
view>)`'s registering units, binds every eval of every unit to the id the registry's own planner mints (`canonical_instruments` ->
`sisteer.registry.jobplan.canonical_instrument` over `load_seed_index(configs/registry)`), and `strict` refuses when any eval mints
no id -- so an eval the seeds do not bind for the demo key is a freeze refusal, never a silent gap. The decode rows and instrument bindings for `qwen35_4b_demo` are
closed on the registry side (tests/test_registry_seeds_demo_instruments.py pins the mirror of the 4B's rows); this module pins what
those bindings add up to for the battery the recipe actually plans, so the two numbers the reproduction plan quotes ("21 evals",
"40 units") are a test and not a recollection of a replay:

* the set of `eval_id`s over the registering rows is the 21 names in :data:`REGISTERING_EVALS`, literally, and each eval is carried
  by exactly the family :data:`EVALS_BY_FAMILY` says;
* the unit count per family is :data:`UNITS_PER_FAMILY` under a one-label placement (40 units: the four attacker-pool members are
  units of the multi-turn group there) and :data:`REGISTERING_UNITS_PER_FAMILY` placement-free (36: the freeze's own call,
  `clusters=None` -- attacker members register nothing and are not units of the inventory; the only difference between the two
  plans is those four members);
* every one of the 21 evals mints a 64-hex `instrument_id` from the seeds dir alone -- the registry root pointed at an empty
  directory for the planner's subprocess -- naming the seed instrument and the demo key's own decode-policy ref of
  :data:`CANONICAL` (the three cap-4,096 multi-turn evals on the demo's `|multi_turn|off|vendor_preset` row, gsm8k on
  `|gsm8k|on|thinking_card` because the table thinks for gsm8k, mmlu on the explicit `|mmlu|off|greedy` row); and the id the
  subprocess reports is the id the in-process planner mints, which is the sha256 over the instrument's canonical identity fields.

The view is read through the same call the freeze makes (`models.baseline_view`: the single-phase recipe's one steered wave), so the
pin follows the models table's declaration of that wave rather than spelling its name; the wave's own name and cells are pinned by
the table's tests. The pin is deliberately independent of the generated configs: `unit_plan` plans from the models table and the
seeds dir, which is what lets this replay run on a tree with no staged bank.
"""
from __future__ import annotations

import collections
import inspect
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import models as M  # noqa: E402
import stage as ST  # noqa: E402
from sisteer.registry import identity as ID  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402

KEY = "qwen35_4b_demo"
TABLE = str(REPO / "reproduce" / "models.yaml")
SEEDS = str(REPO / "configs" / "registry")
SHA = "a" * 40
HEX64 = re.compile(r"^[0-9a-f]{64}$")
#: the composition knobs `unit_plan` reads from the environment when the caller passes none: cleared so the pin is the table's plan,
#: not a site's (a different shard cover changes the multi-turn target count; a seed base changes an env value, never a unit)
PLAN_KNOBS = ("SISTEER_SHARD_GROUPS", "SISTEER_MT_ATTACKER_GPUS", "SISTEER_MT_SEED_BASE", "SISTEER_REGISTRY_SEEDS", "SISTEER_SRC_DIR")

#: the 21 evals the battery registers (the set of `eval_id`s over the registering rows), literally
REGISTERING_EVALS = frozenset({
    "pasttense", "codeattack", "renellm", "grok420",                                   # static (and padding: grok420 alone)
    "strongreject_bon",                                                                # bon: the 512 store and the 4,096 climb
    "crescendo", "pair", "actorattack",                                                # multiturn
    "petri_pooled", "petri_cyber_bio_chem_capabilities", "petri_self_preservation", "petri_delusion_sycophancy",
    "petri_defer_to_users", "petri_reasoning_unfaithfulness", "petri_instruction_evasion",   # petri (one proxy, seven evals)
    "agentic_leaking", "agentic_murder",                                               # agentic
    "xstest", "or_bench_hard",                                                         # controls
    "mmlu",                                                                            # mmlu
    "gsm8k",                                                                           # gsm8k (one unit per cell of the view)
})
#: which family's units carry each eval (a registering row's `family` -> the evals its rows name)
EVALS_BY_FAMILY = {
    "static": {"pasttense", "codeattack", "renellm", "grok420"},
    "padding": {"grok420"},
    "bon": {"strongreject_bon"},
    "multiturn": {"crescendo", "pair", "actorattack"},
    "petri": {"petri_pooled", "petri_cyber_bio_chem_capabilities", "petri_self_preservation", "petri_delusion_sycophancy",
              "petri_defer_to_users", "petri_reasoning_unfaithfulness", "petri_instruction_evasion"},
    "agentic": {"agentic_leaking", "agentic_murder"},
    "controls": {"xstest", "or_bench_hard"},
    "mmlu": {"mmlu"},
    "gsm8k": {"gsm8k"},
}
#: units per family under a ONE-LABEL placement universe (40): multiturn 16 = 4 attacker-pool members + 12 targets (3 attacks x 4
#: shard groups), gsm8k 10 (one per cell of the view), padding 5 (one per level), static 2 (t0 / t1), bon 2 (store512 + climb),
#: controls 2 (t0 / t1), petri 1, agentic 1, mmlu 1
UNITS_PER_FAMILY = {"multiturn": 16, "gsm8k": 10, "padding": 5, "static": 2, "bon": 2, "controls": 2, "petri": 1, "agentic": 1, "mmlu": 1}
#: the REGISTERING units (36): the same minus the four attacker members, which register nothing -- the freeze's placement-free plan
REGISTERING_UNITS_PER_FAMILY = {**UNITS_PER_FAMILY, "multiturn": 12}
ATTACKER_MEMBERS = 4
#: the view's ten cells, in the table's order: the gsm8k family plans one unit per cell, so `gsm8k: 10` above is this list's length
VIEW_CELLS = ["none", "prompted__safety_refusal", "prompted__si_constitution", "refusal_direction", "control_lora", "constitution_lora",
              "difference_lora", "control_relu", "constitution_relu", "difference_relu"]

#: eval -> (seed instrument id, the demo key's decode-policy ref) the registry planner binds for (eval, qwen35_4b_demo, lora_native)
CANONICAL = {
    "pasttense": ("static_sampled_vendor_preset_4096_8192", f"{KEY}|static|off|vendor_preset"),
    "codeattack": ("static_sampled_vendor_preset_4096_8192", f"{KEY}|static|off|vendor_preset"),
    "renellm": ("static_sampled_vendor_preset_4096_8192", f"{KEY}|static|off|vendor_preset"),
    "grok420": ("static_sampled_vendor_preset_4096_8192", f"{KEY}|static|off|vendor_preset"),
    "strongreject_bon": ("bon_512store_4096climb_T1", f"{KEY}|bon|off|grandfathered_T1"),
    # the registry owner's split of 2026-10-14 (origin/main registry-decode-history, merged in integration round 3a): the demo key's multi-turn
    # instruments bind the vendor-preset seeds `multiturn_<attack>_cap4096_vendor_preset` (decode_seed_rule per_goal_turn; shared with the 35B
    # and gemma4_31b), while the paper's `_generation_config` seeds keep the served models' as-filed rows under `none`
    "crescendo": ("multiturn_crescendo_cap4096_vendor_preset", f"{KEY}|multi_turn|off|vendor_preset"),
    "pair": ("multiturn_pair_cap4096_vendor_preset", f"{KEY}|multi_turn|off|vendor_preset"),
    "actorattack": ("multiturn_actorattack_cap4096_vendor_preset", f"{KEY}|multi_turn|off|vendor_preset"),
    "petri_pooled": ("petri_full22_as_run", f"{KEY}|petri|off|vllm_default"),
    "petri_cyber_bio_chem_capabilities": ("petri_full22_as_run", f"{KEY}|petri|off|vllm_default"),
    "petri_self_preservation": ("petri_full22_as_run", f"{KEY}|petri|off|vllm_default"),
    "petri_delusion_sycophancy": ("petri_full22_as_run", f"{KEY}|petri|off|vllm_default"),
    "petri_defer_to_users": ("petri_full22_as_run", f"{KEY}|petri|off|vllm_default"),
    "petri_reasoning_unfaithfulness": ("petri_full22_as_run", f"{KEY}|petri|off|vllm_default"),
    "petri_instruction_evasion": ("petri_full22_as_run", f"{KEY}|petri|off|vllm_default"),
    "agentic_leaking": ("agentic_misalignment_runner_T1_3000_16384", f"{KEY}|agentic|off|runner_forced_T1"),
    "agentic_murder": ("agentic_misalignment_runner_T1_3000_16384", f"{KEY}|agentic|off|runner_forced_T1"),
    "xstest": ("overrefusal_xstest_sampled_vendor_preset_ladder", f"{KEY}|over_refusal|off|vendor_preset"),
    "or_bench_hard": ("overrefusal_orbench_sampled_vendor_preset_ladder", f"{KEY}|over_refusal|off|vendor_preset"),
    "mmlu": ("mmlu_letter_prefill_full_1531", f"{KEY}|mmlu|off|greedy"),
    "gsm8k": ("gsm8k_paired200_inspect_10shot_thinking_on_8192", f"{KEY}|gsm8k|on|thinking_card"),
}


@pytest.fixture
def table_plan(monkeypatch):
    """The battery's plan from the table alone: the composition knobs cleared, the view read as the freeze reads it."""
    for k in PLAN_KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", raising=False)
    view = M.baseline_view(KEY, TABLE)
    assert M.single_phase(KEY, TABLE) and view in M.views(KEY, TABLE) and not M.steered_wave_view(view, KEY, TABLE)
    assert M.model(KEY, TABLE)["views"][view]["cells"] == VIEW_CELLS
    return view


def _by_family(rows):
    return dict(collections.Counter(r["family"] for r in rows))


def test_the_pins_are_consistent_with_each_other():
    assert set(CANONICAL) == REGISTERING_EVALS and len(REGISTERING_EVALS) == 21
    assert set().union(*EVALS_BY_FAMILY.values()) == REGISTERING_EVALS
    assert set(EVALS_BY_FAMILY) == set(UNITS_PER_FAMILY) == set(REGISTERING_UNITS_PER_FAMILY) == set(ST.FAMILIES)
    assert sum(UNITS_PER_FAMILY.values()) == 40 and sum(REGISTERING_UNITS_PER_FAMILY.values()) == 36
    assert UNITS_PER_FAMILY["multiturn"] - REGISTERING_UNITS_PER_FAMILY["multiturn"] == ATTACKER_MEMBERS
    assert UNITS_PER_FAMILY["gsm8k"] == len(VIEW_CELLS) and UNITS_PER_FAMILY["padding"] == len(ST.PADDING_LEVELS)
    assert UNITS_PER_FAMILY["multiturn"] - ATTACKER_MEMBERS == len(ST.MT_ATTACKS) * len(ST.DEFAULT_SHARD_GROUPS)
    assert EVALS_BY_FAMILY["static"] == set(ST.STATIC_EVALS) and EVALS_BY_FAMILY["multiturn"] == set(ST.MT_ATTACKS)
    assert EVALS_BY_FAMILY["petri"] == set(ST.PETRI_EVALS) and EVALS_BY_FAMILY["agentic"] == set(ST.AGENTIC_EVALS)
    assert EVALS_BY_FAMILY["controls"] == set(ST.CONTROL_EVALS)


def test_the_battery_registers_exactly_the_pinned_evals_and_unit_counts(table_plan):
    view = table_plan
    # placement-free: the freeze's own call (`expected_units`: unit_plan(key, baseline view, clusters=None), registering rows only)
    rows = ST.unit_plan(KEY, view, table=TABLE, sha=SHA)
    assert len(rows) == 36 and all(r["registers"] for r in rows), [(r["family"], r["unit"], r["registers"]) for r in rows]
    assert _by_family(rows) == REGISTERING_UNITS_PER_FAMILY
    assert {e for r in rows for e in r["evals"]} == REGISTERING_EVALS
    assert {f: {e for r in rows if r["family"] == f for e in r["evals"]} for f in REGISTERING_UNITS_PER_FAMILY} == EVALS_BY_FAMILY
    assert [r["unit"] for r in rows if r["family"] == "gsm8k"] == VIEW_CELLS          # one gsm8k unit per cell, the table's order
    for r in rows:
        assert r["evals"] and r["env"]["F22_REGISTRY_EVALS"] == ",".join(r["evals"]), r["logical_work_id"]   # the job registers EXACTLY its evals
        assert r["logical_work_id"].startswith(f"{KEY}/{view}/{r['family']}/"), r["logical_work_id"]
    assert len({r["logical_work_id"] for r in rows}) == 36
    # one placement label: the attacker-pool members join the plan as units of the multi-turn group and register nothing -- the 40
    # the plan counts; every registering unit is the placement-free plan's, by identity
    placed = ST.unit_plan(KEY, view, table=TABLE, sha=SHA, clusters=["cluster_a"])
    assert len(placed) == 40 and _by_family(placed) == UNITS_PER_FAMILY
    members = [r for r in placed if not r["registers"]]
    assert len(members) == ATTACKER_MEMBERS and all(r["family"] == "multiturn" and r["kind"] == "attacker" and r["evals"] == [] for r in members)
    assert {r["logical_work_id"] for r in placed if r["registers"]} == {r["logical_work_id"] for r in rows}
    assert {e for r in placed for e in r["evals"]} == REGISTERING_EVALS


def test_every_registering_eval_mints_a_canonical_instrument_from_the_seeds_alone(table_plan, monkeypatch, tmp_path):
    model_id = M.model(KEY, TABLE)["serve"]["checkpoint"]["model_id"]
    assert model_id == KEY and M.model(KEY, TABLE)["thinking"]["gsm8k"] == "on"
    empty = tmp_path / "no-registry"
    empty.mkdir()
    monkeypatch.setenv("SISTEER_REGISTRY_ROOT", str(empty))        # the planner's subprocess inherits it: nothing live to fall back on
    evals = sorted(REGISTERING_EVALS)
    canon = ST.canonical_instruments(model_id, evals, seeds_dir=SEEDS, serving_path=ST.SERVING_PATH)
    assert set(canon) == REGISTERING_EVALS
    errors = {e: c["error"] for e, c in canon.items() if "error" in c}
    assert not errors, errors
    idx = JP.load_seed_index(SEEDS)
    for e in evals:
        c = canon[e]
        assert HEX64.match(c["instrument_id"] or ""), (e, c)
        assert (c["seed_instrument_id"], c["decode_policy_ref"]) == CANONICAL[e], (e, c)
        payload, extra = JP.canonical_instrument(idx, e, model_id, ST.SERVING_PATH)      # the same planner, in process
        assert payload["instrument_id"] == c["instrument_id"] == ID.instrument_id(payload), e
        assert extra.get("seed_instrument_id") == CANONICAL[e][0] and model_id in (extra.get("model_ids") or []), (e, extra)
        assert c["decode_policy_ref"] in [str(r) for r in (payload.get("decode_policy_refs") or extra.get("decode_policy_refs") or [])], (e, c)
    # the three the reproduction plan spells out, verbatim
    assert canon["crescendo"]["decode_policy_ref"] == f"{KEY}|multi_turn|off|vendor_preset"
    assert canon["mmlu"]["decode_policy_ref"] == f"{KEY}|mmlu|off|greedy"
    assert canon["gsm8k"]["decode_policy_ref"] == f"{KEY}|gsm8k|on|thinking_card"
    # the mmlu read is the full 1,531-item instrument the front-door key registers natively
    assert canon["mmlu"]["item_set_id"] and canon["mmlu"]["n_items"] == 1531 and HEX64.match(canon["mmlu"]["items_sha256"] or "")


def test_the_freeze_replays_the_same_plan():
    """The freeze's inventory IS this replay: `expected_units` plans the table's baseline view placement-free, keeps the registering
    rows, hands the set of their evals to `canonical_instruments`, and under `strict` refuses an eval that mints no id by name."""
    src = inspect.getsource(ST.expected_units)
    assert "base_view = M.baseline_view(model_key, table)" in src
    # the freeze plans placement-free (clusters=None) under the overrides document's smoke knob, when one is set (demo-import M10), threads the
    # `rerun-units` stamp's declarations (`reruns`; never the freeze's own), and keeps the registering rows — the call is pinned WHOLE on the
    # whitespace-normalized source, so a wrapped argument list cannot hide a dropped keyword
    assert ("unit_plan(model_key, base_view, table=table, families=families, nshards=nshards, sha=code_sha, clusters=None, smoke_goals=smoke, "
            "reruns=reruns)") in " ".join(src.split())
    assert 'if r["registers"]]' in src and "smoke = smoke_goals(overrides, model_key)" in src
    assert 'all_evals = sorted({e for r in rows for e in r["evals"]})' in src
    assert "canonical_instruments(model_id, all_evals, seeds_dir)" in src
    assert "the registry planner mints no canonical instrument for eval" in src
