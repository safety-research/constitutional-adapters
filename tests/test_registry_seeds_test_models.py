"""The registry seeds of the two paper-companion-repo PIPELINE-TEST models, `qwen35_35b_a3b` and `gemma4_31b`,
whose decode policy comes from the registry. The served gemma id is `gemma4_31b`, with no `_it` suffix.

Every seed a new model needs before its first job is a HARD prerequisite: an unregistered `model_keys` spelling is
refused by self-registration (`common.sh f22_registry_register` -> `jobplan.model_for_key`), a (model, eval, thinking) with no
decode row is refused at launch (`decode_policy.NoPolicy`, never invented), an instrument whose `decode_policy_refs` do not name the
model cannot be bound (`jobplan.canonical_instrument`), and `si_token_count_by_model` is per model. This module pins all of that
for both models against the seeds AS LOADED (the real loader, zero refusals) and against the COMMITTED export the entry points read.

Two facts the shape rests on, so they are asserted rather than assumed: (1) the adapter expands `*` wildcard rows over in_paper /
descoped served models ONLY (`dimensions.served_model_ids`), so a `paper_status: reference` model gets NOTHING from the wildcards
and its mmlu / bigcodebench / bfcl rows are explicit (loader-checked `same_values_as` a served model's expansion); (2) the 35B-A3B's
shipped generation_config.json names exactly temperature / top_p / top_k, so no row of it carries a presence / repetition penalty or
min_p FROM THE FILE (the 4B's card rows are the 4B's) -- with ONE deliberate widening (2026-09-30, ruling
`decode-multiturn-vendor-row-sends-six-keys-2026-09-30`): the `|multi_turn|off|vendor_preset` row of each test model states and SENDS
the three at the engine's own defaults (min_p 0.0 / repetition_penalty 1.0 / presence_penalty 0.0), because the multi-turn TARGET's
registry path (`target_decode.registry_vendor_row`) refuses a vendor row that does not send all six -- the refusal every multi-turn
unit of the u0 wave died on; the decoded sampler is unchanged (tests/test_registry_multiturn_vendor_row_sends_six_keys.py).
And ONE sampling-mode route (2026-09-30, ruling `decode-thinking-card-mapping-pipeline-test-models-2026-09-30`): the GSM8K
front door launches the served driver with `--sampling thinking_card`, which resolves its row through the export's thinking_card
mapping, so each test model maps (model_key, thinking_card) -> its `|gsm8k|on|vendor_preset` row -- the shipped file, the
thinking_card rule's first clause; the Qwen card's thinking rows are recorded on that row and not taken
(tests/test_gsm8k_thinking_card_row_test_models.py). Every other sampling-mode route still refuses for both models.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import decode_policy as DP
from sisteer.registry import harvest as H
from sisteer.registry import jobplan as JP
from sisteer.registry import synth
from sisteer.registry.adapters import dimensions as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
EXPORT = os.path.join(ROOT, "configs", "registry", "decode_policy.json")
PACKAGE_EXPORT = os.path.join(ROOT, "src", "sisteer", "registry", "decode_policy.json")
SAMPLED_YAML = os.path.join(ROOT, "configs", "sdfbase_sampled_decode.yaml")
SHIPPED_PIN = os.path.join(ROOT, "configs", "sdfbase_shipped_generation_config.json")
PROD = synth.producer("tests.registry_seeds_test_models")

RULING = "decode-qwen35-35b-a3b-shipped-generation-config-2026-09-29"
#: the 2026-09-30 amendment: the multi_turn vendor rows send all six sampling keys (the driver's contract); the other rows keep the file's three
RULING_MT = "decode-multiturn-vendor-row-sends-six-keys-2026-09-30"
#: the 2026-09-30 thinking_card mapping: `sampling_mode_policy (<model>, thinking_card)` -> `<model>|gsm8k|on|vendor_preset`; no policy row changed
RULING_TC = "decode-thinking-card-mapping-pipeline-test-models-2026-09-30"
MT_TAIL = "multi_turn|off|vendor_preset"
#: what the multi_turn rows state and send beyond the file's three: the value a request that omits each key resolves to on vLLM
MT_ENGINE_DEFAULTS = {"min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}
PHASE_VIEWS = ("u0", "s1", "s2", "d0p75", "d0p75_relu")           # registered as `<key>_<view>` (common.sh key_tag=$KEY_$VIEW)
#: the front-door model in the public tree keeps the plan's five phase views (the removed model's extra dose rungs left with it)
EXTRA_PHASE_VIEWS = {"gemma4_31b": ()}
def phase_views(served): return (*PHASE_VIEWS, *EXTRA_PHASE_VIEWS[served])

#: served model_id -> (fit model_id, HF repo of the served checkpoint, HF repo of the fit checkpoint, the shipped file's top_k)
MODELS = {     # the front-door model whose decode sources (sampled_decode.yaml, the shipped generation-config pin) are in the public tree
    "gemma4_31b": ("gemma4_31b_base", "google/gemma-4-31B-it", "google/gemma-4-31B", 64),
}
#: the models the REGISTRY seeds (configs/registry, the registry owner's): MODELS plus the one whose decode sources left the public tree
SEEDED_MODELS = set(MODELS) | {"qwen35_35b_a3b"}
#: the pinned revisions (cluster A cohort cache refs/main, read 2026-09-29)
REVISIONS = {
    "gemma4_31b": "842da3794eaa0b77d5f08bae87a17459d91ff475", "gemma4_31b_base": "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89",
}
#: the battery (the frozen inventory) as (eval_id, eval_category AS EXPORTED, thinking_mode, the mode the default row must carry);
#: the export spells the categories in the controlled vocabulary (dimensions.EVAL_CATEGORY_SPELLINGS: seed `static` -> static_attack, `bon` -> best_of_n)
BATTERY = (
    ("pasttense", "static_attack", "off", "vendor_preset"), ("codeattack", "static_attack", "off", "vendor_preset"),
    ("renellm", "static_attack", "off", "vendor_preset"), ("grok420", "static_attack", "off", "vendor_preset"),
    ("strongreject_bon", "best_of_n", "off", "vendor_preset"),
    ("crescendo", "multi_turn", "off", "vendor_preset"), ("pair", "multi_turn", "off", "vendor_preset"),
    ("actorattack", "multi_turn", "off", "vendor_preset"),
    ("petri_pooled", "petri", "off", "vendor_preset"),
    ("agentic_leaking", "agentic", "off", "vendor"), ("agentic_murder", "agentic", "off", "vendor"),
    ("xstest", "over_refusal", "off", "vendor_preset"), ("or_bench_hard", "over_refusal", "off", "vendor_preset"),
    ("gsm8k", "capability", "on", "vendor_preset"),
    ("mmlu", "capability", "off", "greedy"),
    ("bigcodebench", "capability", "off", "protocol"), ("bfcl", "capability", "off", "protocol"),
)
#: the eval ids whose canonical instrument (native path) must bind for the served test model, and the seed id it must be
CANONICAL_INSTRUMENTS = {
    "pasttense": "static_sampled_vendor_preset_4096_8192", "codeattack": "static_sampled_vendor_preset_4096_8192",
    "renellm": "static_sampled_vendor_preset_4096_8192", "grok420": "static_sampled_vendor_preset_4096_8192",
    # 2026-10-14 (the registry owner's ruling): the test models' multi-turn route is the vendor-preset one, seeded per (goal, turn), so its
    # canonical seed is multiturn_*_cap4096_vendor_preset (decode_seed_rule per_goal_turn); the generation_config seeds keep the paper's
    # served models alone (tests/test_registry_decode_history.py)
    "crescendo": "multiturn_crescendo_cap4096_vendor_preset", "pair": "multiturn_pair_cap4096_vendor_preset",
    "actorattack": "multiturn_actorattack_cap4096_vendor_preset",
    "xstest": "overrefusal_xstest_sampled_vendor_preset_ladder", "or_bench_hard": "overrefusal_orbench_sampled_vendor_preset_ladder",
    "petri_pooled": "petri_full22_as_run",
    "agentic_leaking": "agentic_misalignment_runner_vendor_3000_16384", "agentic_murder": "agentic_misalignment_runner_vendor_3000_16384",
    "gsm8k": "gsm8k_paired200_inspect_10shot_thinking_on_8192",
    "mmlu": "mmlu_letter_prefill_full_1531",
    "bigcodebench": "bigcodebench_full_protocol_greedy_4096", "bfcl": "bfcl_full_protocol_T0001_1024",
    # the PAPER's canonical best-of-N -- the grandfathered T = 1.0 512-store + 4,096-climb -- binds the test models too
    "strongreject_bon": "bon_512store_4096climb_T1",
}
#: the ruling the two `<model>|bon|off|grandfathered_T1` rows cite (decode_policy_seed.yaml rulings block)
BON_T1_RULING = "decode-bon-canonical-T1-pipeline-test-models-2026-09-29"


@pytest.fixture(scope="module")
def seeds():
    return D.load_seeds(SEEDS)


@pytest.fixture(scope="module")
def bundle(seeds):
    return D.build_records(seeds, PROD, now="2026-09-29T12:00:00.000Z")


@pytest.fixture(scope="module")
def seed_index():
    return JP.load_seed_index(SEEDS)


@pytest.fixture(scope="module")
def pf():
    return DP.PolicyFile.load(EXPORT)


# --------------------------------------------------------------------------------------------
# models.yaml
# --------------------------------------------------------------------------------------------


def test_the_four_model_rows_are_reference_and_pinned(seeds):
    by_id = {m["model_id"]: m for m in seeds.models}
    for served, (fit, hf_served, hf_fit, top_k) in MODELS.items():
        s, f = by_id[served], by_id[fit]
        assert s["paper_status"] == f["paper_status"] == "reference"
        assert s["checkpoint_kind"] == "instruct" and f["checkpoint_kind"] == "base"
        assert (s["hf_repo"], s["hf_revision"]) == (hf_served, REVISIONS[served])
        assert (f["hf_repo"], f["hf_revision"]) == (hf_fit, REVISIONS[fit])
        assert s["family"] == f["family"] and s["family"] in ("qwen35", "gemma4")
        # the shipped file's sampler keys, verbatim (no penalty / min_p key), and the file's own sha
        assert s["generation_config_present"] == 1
        assert s["generation_config_json"] == {"temperature": 1.0, "top_p": 0.95, "top_k": top_k, "do_sample": True}
        assert len(s["generation_config_file_sha256"]) == 64
        # nothing pre-filled that only stage 02 can measure
        for col in ("chat_template_sha256", "tokenizer_sha256", "chat_template_hash16"):
            assert s[col] == f[col] == "unverified", (served, col)
        # the phase views are producer spellings of the SERVED checkpoint (run.view_legacy), never of the fit one
        assert set(f"{served}_{v}" for v in phase_views(served)) <= set(s["producer_spellings"])
        assert not any(str(x) == served for x in f["producer_spellings"]), "the bare key must map to ONE model (legacy_model_map drops a clash)"
    assert "run_in_this_work: false" in by_id["gemma4_31b"]["notes"] and "run_in_this_work: false" in by_id["gemma4_31b_base"]["notes"]
    # the served gemma id is `gemma4_31b`: no `_it` row exists
    assert "gemma4_31b_it" not in by_id


def test_reference_models_are_not_served_models_so_no_wildcard_reaches_them(seeds, bundle):
    """The shape's premise: `*` rows expand over in_paper / descoped served models only, so every row of the test models is explicit."""
    served = D.served_model_ids(seeds.models)
    assert not (set(MODELS) & set(served))
    for pid, (payload, extra) in bundle.policies.rows.items():
        if payload["model_id"] in MODELS:
            assert not extra.get("wildcard_expanded_from"), pid


# --------------------------------------------------------------------------------------------
# legacy_spellings.yaml model_keys: the key, its phase views, the HF repos
# --------------------------------------------------------------------------------------------


def test_every_spelling_resolves_to_the_served_model_with_base_to_instruct(seed_index):
    rows = seed_index.legacy_model_keys
    for served, (fit, _, _, _) in MODELS.items():
        assert JP.model_for_key(rows, served) == (served, "base_to_instruct")
        for view in phase_views(served):
            assert JP.model_for_key(rows, f"{served}_{view}") == (served, "base_to_instruct"), view
        # the fit model rides the row, and the one fit seed of this work is labelled on every view that serves a fitted object
        for row in rows:
            spellings = [x.strip() for x in str(row["legacy"]).split("|")]
            if served in spellings or any(f"{served}_{v}" in spellings for v in phase_views(served)):
                assert row["fit_model_id"] == fit, row["legacy"]
                if any(f"{served}_{v}" in spellings for v in ("s1", "s2", "d0p75", "d0p75_relu", "battery", *EXTRA_PHASE_VIEWS[served])):
                    assert row.get("fit_replicate_label") == "s1", row["legacy"]
                if any(f"{served}_{v}" in spellings for v in ("d0p75", "d0p75_relu")):
                    assert row.get("strength_nominal") == 0.75, row["legacy"]
                for v, dose in (("d0p5", 0.5), ("d1p25", 1.25)):                     # the difference LoRA's other two rungs
                    if f"{served}_{v}" in spellings:
                        assert row.get("strength_nominal") == dose, row["legacy"]
    # a view nobody registered stays a refusal by name (the table is exact-match, one cited row per spelling)
    with pytest.raises(JP.JobPlanRefused, match="not a legacy model spelling"):
        JP.model_for_key(rows, "qwen35_35b_a3b_s3")
    with pytest.raises(JP.JobPlanRefused, match="not a legacy model spelling"):
        JP.model_for_key(rows, "gemma4_31b_it")


def test_hf_repo_spellings_resolve_in_the_export_and_the_loader_map(seeds, pf):
    """D's trainer maps a sidecar's HF repo through `PolicyFile.model_id` (the export's `model_spellings`); the loader's
    `legacy_model_map` is the same map at seed time. Both name exactly the four checkpoints."""
    mmap = D.legacy_model_map(seeds)
    for served, (fit, hf_served, hf_fit, _) in MODELS.items():
        assert pf.model_id(hf_served) == mmap[hf_served] == served
        assert pf.model_id(hf_fit) == mmap[hf_fit] == fit
        assert pf.model_id(served) == served and pf.model_id(fit) == fit
        for view in phase_views(served):
            assert pf.model_id(f"{served}_{view}") == mmap[f"{served}_{view}"] == served
    assert pf.models["qwen35_35b_a3b"]["hf_repo"] == "Qwen/Qwen3.5-35B-A3B" and pf.models["gemma4_31b"]["hf_repo"] == "google/gemma-4-31B-it"
    assert pf.models["qwen35_35b_a3b"]["paper_status"] == pf.models["gemma4_31b"]["paper_status"] == "reference"


# --------------------------------------------------------------------------------------------
# decode_policy_seed.yaml: ten rows per model, from the shipped file; the export resolves every battery setting
# --------------------------------------------------------------------------------------------


def test_ruling_is_in_the_seed_and_every_new_row_cites_a_ruling(seeds, bundle):
    assert RULING in seeds.decode["rulings"] and RULING_MT in seeds.decode["rulings"] and RULING_TC in seeds.decode["rulings"]
    idx = bundle.policies
    for pid, (payload, extra) in idx.rows.items():
        if payload["model_id"] not in MODELS:
            continue
        if payload["policy_ref"].endswith("|bon|off|grandfathered_T1"):
            # the canonical T1 climb's row -- non-default; superseded by the model's vendor row until 2026-10-14, LIVE since the
            # ruling of that day (decode-bon-canonical-T1-live-non-default-2026-10-14), as the paper models' grandfathered rows are
            assert extra.get("decode_ruling_ref") == BON_T1_RULING, pid
            assert payload["is_default"] == 0 and payload["superseded_by_policy_id"] is None, pid
            assert "decode-bon-canonical-T1-live-non-default-2026-10-14" in str(payload.get("notes")), pid
        elif payload["policy_ref"].endswith("|" + MT_TAIL):
            # 2026-09-30: the multi-turn TARGET row cites the six-keys amendment (an in-place edit of @v1; still THE default, not superseded)
            assert extra.get("decode_ruling_ref") == RULING_MT, pid
            assert payload["is_default"] == 1 and not payload.get("superseded_by_policy_id"), pid
        else:
            assert extra.get("decode_ruling_ref") in (RULING, "decode-protocol-rows-2026-09-11"), pid
            assert payload["is_default"] == 1 and not payload.get("superseded_by_policy_id"), pid
        assert payload["effective_from_utc"] == "2026-09-29T00:00:00.000Z", pid
    # exactly eleven rows per model, and the refs they must spell
    for served in MODELS:
        refs = sorted(payload["policy_ref"] for payload, _ in idx.rows.values() if payload["model_id"] == served)
        assert refs == sorted(f"{served}|{tail}" for tail in (
            "static|off|vendor_preset", "over_refusal|off|vendor_preset", "bon|off|vendor_preset", "bon|off|grandfathered_T1", "multi_turn|off|vendor_preset",
            "petri|off|vendor_preset", "agentic|off|vendor", "gsm8k|on|vendor_preset", "mmlu|off|greedy",
            "bigcodebench|off|protocol", "bfcl|off|protocol")), served


def test_vendor_rows_state_exactly_the_shipped_files_keys(bundle, seeds):
    """A row carries EXACTLY the keys its source names: the 35B-A3B's file has no penalty / min_p key, so no row of it sends one
    (the 4B's card rows with presence_penalty 1.5 are the 4B's); gemma-4-31B-it's file equals the E4B's. ONE widening
    (2026-09-30, RULING_MT): the multi-turn TARGET row additionally states and SENDS min_p / repetition_penalty / presence_penalty at the
    engine's own defaults -- `target_decode.registry_vendor_row` (the driver's registry path) refuses a vendor row that does not send all
    six, and the ingest matcher can match a six-key-sending run only against a row that pins the six; the file's three numbers are
    unchanged and the row no longer claims `same_values_as` the static row (the loader compares all six stated numbers)."""
    idx = bundle.policies
    for served, (_, _, _, top_k) in MODELS.items():
        for tail in ("static|off|vendor_preset", "over_refusal|off|vendor_preset", "bon|off|vendor_preset", MT_TAIL,
                     "petri|off|vendor_preset", "agentic|off|vendor", "gsm8k|on|vendor_preset"):
            pid = f"{served}|{tail}@v1"
            payload, extra = idx.rows[pid]
            assert payload["decode_source"] == "generation_config", pid
            sent = set(extra["params_sent"])
            if tail == MT_TAIL:
                assert idx.numbers(pid) == {"temperature": 1.0, "top_p": 0.95, "top_k": top_k, **MT_ENGINE_DEFAULTS}, pid
                assert {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "max_tokens", "seed"} <= sent, pid
                assert set(payload["params_not_sent_json"]) == {"frequency_penalty"}, pid
                assert "same_values_as" not in extra, pid
                continue
            assert idx.numbers(pid) == {"temperature": 1.0, "top_p": 0.95, "top_k": top_k, "min_p": None,
                                        "repetition_penalty": None, "presence_penalty": None}, pid
            assert {"temperature", "top_p", "top_k"} <= sent and not ({"presence_penalty", "repetition_penalty", "min_p"} & sent), pid
            assert set(payload["params_not_sent_json"]) == {"min_p", "repetition_penalty", "presence_penalty", "frequency_penalty"}, pid
            if tail != "static|off|vendor_preset":
                assert extra.get("same_values_as") == f"{served}|static|off|vendor_preset@v1", pid
    # The drivers' SAMPLED route (configs/sdfbase_sampled_decode.yaml presets) and the shipped generation-config pin
    # (configs/sdfbase_shipped_generation_config.json) ship decode sources for the paper's served models and the worked example only:
    # the public tree dropped the pipeline-test models' presets and pins (#478 / #482) while the REGISTRY keeps their rows as the paper's
    # record. So the two tables are compared against the seeds restricted to what the public tree ships -- every shipped key resolves to a
    # seeded model that carries its static vendor row (a shipped key with no seed row fails by name) -- and the registry-only models are
    # asserted by name: seeded above, absent from both tables.
    import yaml
    presets = yaml.safe_load(open(SAMPLED_YAML, encoding="utf-8"))["presets"]
    pin = json.load(open(SHIPPED_PIN, encoding="utf-8"))["models"]
    registry_only = {served: hf for served, (_, hf, _, _) in MODELS.items()} | {"qwen35_35b_a3b": "Qwen/Qwen3.5-35B-A3B"}
    assert set(registry_only) == SEEDED_MODELS
    for served, hf_served in registry_only.items():
        assert f"{served}|static|off|vendor_preset@v1" in idx.rows and served not in presets and hf_served not in pin, served
    mmap = D.legacy_model_map(seeds)
    shipped = {key: mmap.get(key) for key in presets} | {hf: mmap.get(hf) for hf in pin}
    assert all(shipped.values()), {k: v for k, v in shipped.items() if not v}                    # every shipped key names a seeded model
    for key, mid in sorted(shipped.items()):
        assert f"{mid}|static|off|vendor_preset@v1" in idx.rows, (key, mid)                       # ...that carries its static vendor row
    assert {mmap[k] for k in presets} == {mmap[hf] for hf in pin} | {"qwen35_4b_demo"}, (sorted(presets), sorted(pin))   # the two tables agree
    for key, p in presets.items():                                                                # a preset states the numbers its seed row sends
        row, extra = idx.rows[f"{mmap[key]}|static|off|vendor_preset@v1"]
        stated = {k: v for k, v in idx.numbers(f"{mmap[key]}|static|off|vendor_preset@v1").items() if k in extra["params_sent"]}
        assert {k: float(v) for k, v in p["preset"].items() if k in stated} == {k: float(v) for k, v in stated.items() if k in p["preset"]}, key


def test_explicit_wildcard_shaped_rows_equal_the_served_models_expansions(bundle):
    idx = bundle.policies
    for served in MODELS:
        for tail, mode, src in (("mmlu|off|greedy", "greedy", "greedy"), ("bigcodebench|off|protocol", "protocol", "greedy"),
                                ("bfcl|off|protocol", "protocol", "greedy")):
            pid = f"{served}|{tail}@v1"
            payload, extra = idx.rows[pid]
            twin = f"gemma4e4b_it|{tail}@v1"
            assert extra["same_values_as"] == twin and idx.numbers(pid) == idx.numbers(twin), pid
            assert extra["mode"] == mode and payload["decode_source"] == src, pid
            assert extra["params_sent"] == idx.rows[twin][1]["params_sent"], pid
            assert idx.rows[twin][1]["wildcard_expanded_from"] == f"*|{tail}"      # the twin IS a wildcard expansion; ours is not


@pytest.mark.parametrize("served", sorted(MODELS))
def test_the_export_resolves_every_battery_setting_without_a_wildcard(pf, served):
    """`decode_policy.resolve` (the launch-time authority) returns ONE row per battery eval x thinking mode for the served test model,
    the row is the model's own (never a `*` expansion), and its mode is the ruled one."""
    top_k = MODELS[served][3]
    for eval_id, category, thinking, mode in BATTERY:
        pol = pf.resolve(served, eval_id=eval_id, thinking_mode=thinking)
        assert pol.model_id == served and pol.mode == mode, (eval_id, pol.policy_id)
        assert not pf.by_id[pol.policy_id].get("wildcard_expanded_from"), (eval_id, pol.policy_id)   # the exported row, not a `*` expansion
        assert pol.is_default == 1 and pol.superseded_by_policy_id is None
        if mode in ("vendor_preset", "vendor"):
            # the multi-turn TARGET row sends the six (RULING_MT); every other vendor row sends exactly the file's three
            extra_sent = MT_ENGINE_DEFAULTS if category == "multi_turn" else {}
            assert pol.request_params() == {"temperature": 1.0, "top_p": 0.95, "top_k": top_k, **extra_sent}, (eval_id, pol.policy_id)
            # the category rows resolve for the category too (the harness asks by category for static / controls / multi-turn / petri / agentic)
            if pol.eval_id is None:
                assert pf.resolve(served, eval_category=category, thinking_mode=thinking).policy_id == pol.policy_id
        elif mode == "greedy":
            assert pol.request_params() == {} and pol.numbers()["temperature"] == 0.0
        else:
            assert pol.eval_id == eval_id and pol.numbers()["top_p"] == 1.0 and pol.numbers()["top_k"] == 0
    # the exact refs the harness entry points resolve
    assert pf.by_ref(f"{served}|agentic|off|vendor@v1").mode == "vendor"                         # runner.vendor_policy_ref (exact ref)
    assert pf.by_ref(f"{served}|multi_turn|off|vendor_preset@v1").mode == "vendor_preset"      # target_decode.vendor_preset_policy_ref
    # stage 06 passes F22_MT_TARGET_POLICY_REF=<served>|multi_turn|off|vendor_preset on every multi-turn target (A, 2026-09-29): the bare ref and
    # the category lookup name the same row
    mt = pf.by_ref(f"{served}|multi_turn|off|vendor_preset")
    assert mt.policy_id == f"{served}|multi_turn|off|vendor_preset@v1" == pf.resolve(served, eval_category="multi_turn", thinking_mode="off").policy_id
    assert mt.request_params() == {"temperature": 1.0, "top_p": 0.95, "top_k": top_k, **MT_ENGINE_DEFAULTS}     # the six the driver requires
    assert pf.resolve(served, eval_id="strongreject_bon").policy_ref == f"{served}|bon|off|vendor_preset"   # bon_served.bon_policy
    # the model-independent readers still agree across every model, the explicit rows included
    assert pf.protocol_row("inspect/bfcl_200").numbers()["temperature"] == 0.001
    assert pf.protocol_row("inspect/bigcodebench_200").numbers()["temperature"] == 0.0
    assert pf.model_independent(eval_id="mmlu", mode="greedy", models=[served, "qwen35_4b_it"]).numbers()["temperature"] == 0.0
    # what is NOT seeded is refused by name, never invented: GSM8K thinking-OFF, IFEval, MATH-500, every sampling-mode route
    # but thinking_card
    for kw in (dict(eval_id="gsm8k", thinking_mode="off"), dict(eval_id="ifeval"), dict(eval_id="math500", thinking_mode="on"),
               dict(eval_category="capability", thinking_mode="off"), dict(eval_id="em8q")):
        with pytest.raises(DP.NoPolicy):
            pf.resolve(served, **kw)
    for mode in ("card", "general", "battery", "vendor_preset", "greedy"):
        with pytest.raises(DP.NoPolicy):
            pf.resolve(served, mode=mode)
    # the ONE sampling-mode route seeded for the test models (RULING_TC, 2026-09-30): the GSM8K front door's `--sampling thinking_card`
    # resolves to the model's gsm8k vendor row -- the same row the eval lookup names, the shipped file's three keys
    tc = pf.resolve(served, mode="thinking_card")
    assert tc.policy_id == f"{served}|gsm8k|on|vendor_preset@v1" == pf.resolve(served, eval_id="gsm8k", thinking_mode="on").policy_id
    assert tc.request_params() == {"temperature": 1.0, "top_p": 0.95, "top_k": top_k}


def test_both_committed_exports_carry_the_rows_and_agree():
    with open(EXPORT, encoding="utf-8") as fh, open(PACKAGE_EXPORT, encoding="utf-8") as gh:
        a, b = json.load(fh), json.load(gh)
    assert a == b, "the two committed copies of decode_policy.json differ (run `registry decode-policy export`)"
    refs = {p["policy_ref"] for p in a["policies"]}
    for served in MODELS:
        assert {f"{served}|static|off|vendor_preset", f"{served}|gsm8k|on|vendor_preset", f"{served}|mmlu|off|greedy",
                f"{served}|agentic|off|vendor"} <= refs
    assert a["header"]["policy_set_sha256"] == DP.policy_set_sha256(a)


# --------------------------------------------------------------------------------------------
# instruments.yaml: exactly ONE canonical instrument per battery eval binds for the served test model on the native path
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("served", sorted(MODELS))
def test_exactly_one_canonical_instrument_per_battery_eval(seed_index, served):
    for eval_id, seed_id in CANONICAL_INSTRUMENTS.items():
        payload, extra = JP.canonical_instrument(seed_index, eval_id, served, "lora_native")
        assert extra["seed_instrument_id"] == seed_id, (eval_id, extra["seed_instrument_id"])
        assert served in extra["model_ids"], (eval_id, extra["model_ids"])
        # the bound instrument's decode policy is the model's own row
        pids = extra.get("decode_policy_ids") or [extra.get("decode_policy_id")]
        assert any(str(p).startswith(served + "|") for p in pids), (eval_id, pids)
    # the Petri behaviours ride the pooled instrument row set too
    for eval_id in ("petri_self_preservation", "petri_instruction_evasion"):
        payload, extra = JP.canonical_instrument(seed_index, eval_id, served, "lora_native")
        assert extra["seed_instrument_id"] == "petri_full22_as_run"


@pytest.mark.parametrize("served", sorted(MODELS))
def test_best_of_n_binds_the_papers_canonical_T1_climb_for_the_test_models(seed_index, served, bundle):
    """2026-09-29 (closes the 2026-09-29 KNOWN-OPEN pin): the test models follow the PAPER's canonical best-of-N --
    the grandfathered T = 1.0 512-store + 4,096-climb instrument `bon_512store_4096climb_T1`, the shape reproduce/06_eval_waves.sh submits
    (target_static.sbatch families=bon + bon_climb.sbatch). Each model has a `<model>|bon|off|grandfathered_T1` row mirroring qwen35_4b_it's
    (temperature 1.0 sent, top_p / top_k from the shipped file, non-default, superseded by the vendor row for new draws), referenced from that
    instrument, so job-start registration of a BoN run binds a canonical instrument. The vendor-preset BoN instruments stay `sensitivity`
    (r2v1-include-qwen-bon-vendor_preset-new-instrument is not overruled), and the vendor row stays THE default (bon_served.bon_policy)."""
    payload, extra = JP.canonical_instrument(seed_index, "strongreject_bon", served, "lora_native")
    assert extra["seed_instrument_id"] == "bon_512store_4096climb_T1" and served in extra["model_ids"]
    assert f"{served}|bon|off|grandfathered_T1@v1" in extra["decode_policy_ids"]
    assert payload["decode_source"] == "grandfathered_T1" and payload["decode_temperature"] == 1.0 and payload.get("decode_top_p") is None
    # the paper's five served models still ride the same row
    for mid in ("gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash"):
        assert mid in extra["model_ids"], mid
    # the vendor-preset BoN instruments stay sensitivity: the native-4096 one carries the test models' refs (C-E's seed), the
    # 512-store + climb vendor one carries the paper's five only -- neither is canonical for anyone
    native = [(p, e) for p, e in seed_index.instruments if e.get("seed_instrument_id") == "bon_native4096_vendor_preset" and served in (e.get("model_ids") or [])]
    assert len(native) == 1 and JP._paper_status(native[0][0], native[0][1]) == "sensitivity"
    for p, e in seed_index.instruments:
        if e.get("seed_instrument_id") in ("bon_native4096_vendor_preset", "bon_512store_4096climb_vendor_preset"):
            assert JP._paper_status(p, e) == "sensitivity", e.get("seed_instrument_id")
    # the T1 row: the paper's shape, numerically the vendor row for these models, readable by reference; superseded by the vendor row until
    # 2026-10-14 and LIVE (non-default) since the rule of that day -- canonical best-of-N stays this T = 1 instrument
    t1, t1x = bundle.policies.rows[f"{served}|bon|off|grandfathered_T1@v1"]
    vend, _ = bundle.policies.rows[f"{served}|bon|off|vendor_preset@v1"]
    assert t1["decode_source"] == "grandfathered_T1" and t1["is_default"] == 0 and t1["superseded_by_policy_id"] is None
    assert t1x["params_sent"] == ["temperature", "max_tokens", "seed"] and bundle.policies.numbers(t1["policy_id"])["temperature"] == 1.0
    assert bundle.policies.numbers(t1["policy_id"])["top_p"] is None and vend["is_default"] == 1
    assert t1x.get("decode_ruling_ref") == BON_T1_RULING            # the loader refuses a ruling id absent from the seed's rulings block


def test_the_served_models_canonical_instruments_are_unchanged(seed_index):
    """The new refs bind the test models only: the paper's served models keep their instruments (the T1 agentic row, the 200-item heads)."""
    payload, extra = JP.canonical_instrument(seed_index, "agentic_murder", "qwen35_4b_it", "lora_native")
    assert extra["seed_instrument_id"] == "agentic_misalignment_runner_T1_3000_16384"
    for eval_id in ("pasttense", "crescendo", "xstest", "petri_pooled", "gsm8k"):
        JP.canonical_instrument(seed_index, eval_id, "qwen35_4b_it", "lora_native")
    # the vendor agentic sibling names the registry's two onboarded models and nobody else
    vend = [e for _, e in seed_index.instruments if e.get("seed_instrument_id") == "agentic_misalignment_runner_vendor_3000_16384"]
    assert vend and all(set(e["model_ids"]) <= SEEDED_MODELS for e in vend)


def test_gemma_31b_rows_merge_into_the_e4b_instrument_identities(bundle):
    """gemma-4-31B-it's shipped file is byte-identical to the E4B's, so where the instrument identity hashes the sent numbers the two
    models share ONE instrument row (extra.model_ids lists both); the 35B-A3B's 1.0 / 0.95 / 20 matches no served model and is its own."""
    recs = {r["payload"]["instrument_id"]: r["payload"] for r in bundle.records["instrument"]}
    static = [recs[i] for i in bundle.instrument_ids["static_sampled_vendor_preset_4096_8192"]]
    assert len(static) == 24
    for p in static:
        ids = p["extra"]["model_ids"]
        if "gemma4_31b" in ids:
            assert "gemma4e4b_it" in ids and p["decode_top_k"] == 64
        if "qwen35_35b_a3b" in ids:
            assert ids == ["qwen35_35b_a3b"] and p["decode_top_k"] == 20 and p["decode_presence_penalty"] is None


# --------------------------------------------------------------------------------------------
# si_texts.yaml + the whole seed set through the real loader
# --------------------------------------------------------------------------------------------


def test_si_token_counts_carry_an_unverified_key_per_test_model(seeds):
    per_model = 0
    for row in seeds.si_texts:
        counts = row["si_token_count_by_model"]
        if not isinstance(counts, dict):
            # the reference / candidate texts carry ONE scalar `unverified` for every model (si_texts.yaml reference_texts / candidate_texts)
            assert counts == "unverified", row["si_id"]
            continue
        per_model += 1
        for served in MODELS:
            assert counts[served] == "unverified", (row["si_id"], served)
    assert per_model == 4        # the four paper SI texts carry per-model counts


def test_seeds_load_into_a_fresh_registry_with_zero_refusals(tmp_path):
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    bundle, paths = D.write_seed_records(root, SEEDS, producer=PROD, now="2026-09-29T12:00:00.000Z")
    assert paths
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    counts = bundle.counts()
    # 24 models since 2026-10-08: + the worked example's qwen35_4b_demo / qwen35_4b_demo_base (tests/test_registry_seeds_demo_model.py)
    # 190 decode rows since 2026-10-09: 163 (161 + the two canonical-T1 BoN rows) + the worked example's 27 twins of the qwen35_4b_it rows
    # (ruling decode-qwen35-4b-demo-mirrors-4b-rows-2026-10-09; tests/test_registry_seeds_demo_instruments.py)
    assert counts["model"] == 24 and counts["decode_policy"] == 191     # + the worked example's explicit MMLU greedy row (2026-10-09)
    assert bundle.policies.no_default_keys == []
    assert not [w for w in bundle.warnings if "35b" in w or "gemma4_31b" in w], bundle.warnings
