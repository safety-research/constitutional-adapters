"""The Qwen seed-replicate native battery: the planned-run identity files.

Generated against a FIXTURE decode-policy export = the committed export + the three vendor rows that are landing
(``bon`` / ``multi_turn`` / ``petri`` ``|off|vendor_preset@v1``), because the rows are not on this base yet; the W gate
checks the real registry. Every assertion reads CONTENT (ids, shas, keys, values) off the generated
files, and every check is exercised against a KNOWN-BAD input as well as the good one.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase" / "seedrep",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import registry_plans as RP  # noqa: E402
from sisteer.registry import emit, identity  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import validate as VAL  # noqa: E402
from sisteer.registry.decode_policy import PolicyFile  # noqa: E402

W = "0c19251100000000000000000000000000000000"
CAMPAIGN = "seedrep_battery_test"
REAL_POLICY = REPO / "configs" / "registry" / "decode_policy.json"
NEW_ROWS = {"bon": ("strongreject_bon", "best_of_n", "per_draw", "4096"),
            "multi_turn": (None, "multi_turn", "per_item", "4096 per reply"),
            "petri": (None, "petri", "per_item", "1024 per turn")}


def fixture_policy(tmp_path: Path, *, superseded: bool = False, drop: str = "", petri_seed_rule: str = None, agentic_seed_rule: str = None) -> Path:
    """The committed export with the three vendor rows AS THIS FIXTURE SPELLS THEM (they were pending once; main cc994f18 has
    since landed them in the export, so the fixture REPLACES the export's copies rather than appending beside them — appending made
    ``by_ref`` refuse "names 2 rows"); ``superseded`` marks the BoN row superseded, ``drop`` omits one row (removed from the export
    too) — the known-bad variants. ``petri_seed_rule`` / ``agentic_seed_rule`` set the ROW's ``seed_rule`` (the
    ``registry-seed-rules`` export state: ``per_scenario_turn`` / ``per_response``). Every field the planner reads off a row
    (policy_id, decode_source, request_params, model_id, superseded_by_policy_id, seed_rule) is the fixture's; the real export's rows
    carry the landed labels (petri ``per_scenario_turn``, multi_turn ``per_goal_turn``), so a plan built here differs from one built on
    ``REAL_POLICY`` exactly where a test expects it to."""
    d = json.loads(REAL_POLICY.read_text())
    tmpl = next(p for p in d["policies"] if p["policy_id"] == "qwen35_4b_it|static|off|vendor_preset@v1")
    fixture_ids = {f"qwen35_4b_it|{cat}|off|vendor_preset@v1" for cat in NEW_ROWS}
    d["policies"] = [p for p in d["policies"] if p["policy_id"] not in fixture_ids]     # the export's own copies (main #216) give way
    for cat, (eval_id, ecat, seed_rule, cap) in NEW_ROWS.items():
        if cat == drop:
            continue
        r = copy.deepcopy(tmpl)
        ref = f"qwen35_4b_it|{cat}|off|vendor_preset"
        r.update({"policy_id": ref + "@v1", "policy_ref": ref, "policy_version": 1, "eval_id": eval_id, "eval_category": ecat,
                  "seed_rule": petri_seed_rule if (cat == "petri" and petri_seed_rule) else seed_rule, "max_new_tokens_rule": cap, "is_default": 1,
                  "superseded_by_policy_id": None, "source_citation": "TEST FIXTURE of the D9 row", "notes": "fixture"})
        if cat == "bon" and superseded:
            r["superseded_by_policy_id"] = "qwen35_4b_it|bon|off|vendor_preset@v2"
        d["policies"].append(r)
    if agentic_seed_rule:
        next(p for p in d["policies"] if p["policy_id"] == "qwen35_4b_it|agentic|off|vendor@v1")["seed_rule"] = agentic_seed_rule
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / f"decode_policy_fixture{'_sup' if superseded else ''}{'_no_' + drop if drop else ''}{'_' + petri_seed_rule if petri_seed_rule else ''}.json"
    out.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n")
    return out


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    pol = fixture_policy(tmp_path_factory.mktemp("policy"))
    return RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=pol)


@pytest.fixture(scope="module")
def plan_set(inputs):
    return RP.build(inputs)


@pytest.fixture(scope="module")
def plans_dir(plan_set, tmp_path_factory):
    out = tmp_path_factory.mktemp("plans")
    RP.write_plan_set(plan_set, out)
    return out


def _pins():
    return json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text())["qwen35_4b"]


def test_plan_set_shape_run_ids_unique_and_every_record_validates(plan_set, plans_dir):
    idx = plan_set.index
    assert idx["n_plan_files"] == 405 and idx["n_store_files"] == 27 and idx["n_run_ids"] == 540
    assert set(idx["legs"]) == set(RP.LEG_NAMES) and len(RP.LEG_NAMES) == 15
    files = list(RP.iter_plan_files(plans_dir))
    assert len(files) == 405
    rids = plan_set.run_ids()
    assert len(rids) == 540 and len(set(rids)) == 540
    # 27 arm-runs per leg: 15 fitted (5 x 3 seeds) + 3 none + 6 prompted + 3 Arditi
    for leg in RP.LEG_NAMES:
        arms = idx["legs"][leg]["arms"]
        kinds = [a[str(s)]["arm_kind"] for a in arms.values() for s in RP.SEEDS if str(s) in a]
        assert len(kinds) == 27 and kinds.count("fitted") == 15 and kinds.count("none") == 3 and kinds.count("prompted") == 6 and kinds.count("arditi") == 3
        objs = {a[str(s)]["object_id"] for a in arms.values() for s in RP.SEEDS if str(s) in a}
        assert len(objs) == 15 + 1 + 2 + 1            # 15 pins, `none`, two `si:` objects, the Arditi sha
    # every identity, as the plan file carries it, validates STRICTLY through the real record validator
    producer = emit.producer_block("tests.synthetic")
    for pf in files:
        for route, ident in pf.identities.items():
            ids, run_payload, state_payload = emit.planned_run_records(ident, producer)
            assert ids == ident["run_id"]
            validated = {}
            for rt, payload in (("run", run_payload), ("run_state", state_payload)):
                rec = R.make_record(rt, payload, producer, identity.producer_record_id(), R.utc_now())
                v = VAL.validate_record(rec, str(pf.path))
                assert isinstance(v, VAL.Validated), (pf.path, route, v)
                validated[rt] = v
            run_v = validated["run"]
            assert run_v.extra.get("seed_rule") == "literal_training_seed"       # the lane facts ride in extra_json, never dropped
            assert run_v.payload["seed_base"] == pf.s and run_v.payload["decode_policy_id"] == ident["decode_policy_id"]
            assert run_v.payload["decode_source"] in ("model_card", "greedy", "vendor_recommendation")


def test_fitted_identity_content_pins_seed_and_exact_policy_row(plans_dir, inputs):
    pf = RP.load_plan_file(plans_dir / RP.plan_relpath("pasttense", "constv2-mix-full__psr_ce_all", 2))
    (route, ident), = pf.identities.items()
    assert route == "*"
    pin = _pins()["base64s2__constv2-mix-full__psr_ce_all"]["object_sha256"]
    assert ident["object_id"] == pin and ident["producer_object_key_legacy"] == "base64s2__constv2-mix-full__psr_ce_all"
    assert ident["producer_run_key_legacy"] == f"seedrep_battery:{W[:7]}:pasttense:s2"
    assert ident["seed_base"] == 2 and ident["extra"]["seed_rule"] == "literal_training_seed" and ident["extra"]["seed_rule_detail"] is None
    assert ident["decode_policy_id"] == "qwen35_4b_it|static|off|vendor_preset@v1" and ident["extra"]["decode_policy_ref"] == ident["decode_policy_id"]
    pol = inputs.policy.by_ref("qwen35_4b_it|static|off|vendor_preset@v1")
    assert ident["decode_source"] == pol.decode_source == "model_card" and pol.superseded_by_policy_id is None
    assert ident["decode_seed_rule"] == "per_item" and ident["thinking_mode"] == "off" and ident["token_cap_tokens"] == 4096
    assert ident["token_cap_ladder_json"] == [4096, 8192, 16384] and ident["context_window_tokens"] == 11520
    assert ident["transfer_kind"] == "base_to_instruct" and ident["applied_positions_mode"] == "all" and ident["multi_turn_steering_scope"] == "single_turn"
    assert ident["model_id"] == "qwen35_4b_it" and ident["code_sha"] == W and ident["substrate"] == "modal" and ident["substrate_source"] == "recorded"
    assert ident["judge_id"] == inputs.judge_ids["strongreject_v1_haiku45"] and len(ident["judge_id"]) == 64
    assert ident["extra"]["served_key"] == "constv2-mix-full__psr_ce_all" and ident["extra"]["view"] == "s2" and ident["extra"]["store"] == "static"
    assert ident["run_id"] == identity.run_id(CAMPAIGN, ident["producer_run_key_legacy"], ident["producer_object_key_legacy"], "pasttense", ident["instrument_id"])
    # the instrument is MINTED from the exported field set and differs from the canonical seed instrument it derives from
    inst = json.loads((plans_dir / RP.INSTRUMENTS_DIR / f"{ident['instrument_id']}.json").read_text())
    assert identity.instrument_id(inst["fields"]) == ident["instrument_id"]
    assert inst["provenance"]["derived_from_seed_instrument"] == "static_sampled_vendor_preset_4096_8192"
    assert inst["provenance"]["derived_from_instrument_id"] != ident["instrument_id"]
    assert inst["fields"]["n_trials"] == 1 and inst["fields"]["decode_presence_penalty"] == 1.5 and inst["fields"]["decode_seed_rule"] == "per_item"
    assert inst["fields"]["context_window_tokens"] == 11520 and inst["fields"]["judge_id"] == ident["judge_id"]
    assert set(inst["provenance"]["changed_fields"]) >= {"n_trials", "harness_version", "context_window_tokens", "token_cap_tokens"}


def test_none_prompted_and_arditi_identities(plans_dir, inputs):
    none3 = RP.load_plan_file(plans_dir / RP.plan_relpath("bon", "none", 3)).identities["*"]
    assert none3["object_id"] == "none" and none3["producer_object_key_legacy"] == "none" and none3["transfer_kind"] == "none"
    assert none3["applied_positions_mode"] == "none" and none3["multi_turn_steering_scope"] == "not_applicable"
    assert none3["decode_seed_rule"] == "per_draw" and none3["decode_policy_id"] == "qwen35_4b_it|bon|off|vendor_preset@v1" and none3["seed_base"] == 3
    assert none3["producer_run_key_legacy"] == f"seedrep_battery:{W[:7]}:bon:s3" and "strength_nominal" not in none3
    si_sha = inputs.si_sha["si_constitution"]
    pr = RP.load_plan_file(plans_dir / RP.plan_relpath("gsm8k", "prompted__si_constitution", 1)).identities["*"]
    assert pr["object_id"] == f"si:{si_sha}" and pr["transfer_kind"] == "prompted_only" and pr["applied_positions_mode"] == "none"
    assert pr["thinking_mode"] == "on" and pr["decode_policy_id"] == "qwen35_4b_it|gsm8k|on|thinking_card@v1" and pr["extra"]["served_key"] == "prompted__si_constitution"
    ar = {s: RP.load_plan_file(plans_dir / RP.plan_relpath("pair", "arditi_allpos", s)).identities["*"] for s in RP.SEEDS}
    arditi_pin = json.loads((REPO / "configs" / "arditi_allpos_pins.json").read_text())["qwen35_4b"]["ar_c0p3__arditi_allpos"]["object_sha256"]
    for s, ident in ar.items():
        assert ident["object_id"] == arditi_pin and ident["producer_object_key_legacy"] == "ar_c0p3__arditi_allpos"
        assert ident["producer_run_key_legacy"] == f"seedrep_battery:{W[:7]}:pair:arditi_s{s}" and ident["seed_base"] == s
        assert ident["extra"]["view"] == "arditi" and ident["extra"]["dose"] == {"c": 0.3, "unit": "raw_coefficient"}
        assert ident["strength_nominal"] == 0.3 and ident["is_operating_point"] == 1 and ident["extra"]["selection_outcome"] == "user_override_over_budget_2026-09-16"
        assert ident["transfer_kind"] == "instruct_direct" and ident["multi_turn_steering_scope"] == "every_turn_prefill_and_decode"
        # the emitted spelling follows the vocabulary in force (main 934750dd / #212 spells per_goal_turn: direct, detail None; a tree
        # without it: per_item + detail) — decided by the planner's own rule, never typed here
        mt_rule, mt_detail = RP.seed_rule_of(RP.MULTITURN_ROW_SEED_RULE, inputs.seed_vocab)
        assert (ident["decode_seed_rule"], ident["extra"]["seed_rule_detail"]) == (mt_rule, mt_detail) == ("per_goal_turn", None)
        assert ident["decode_policy_id"] == "qwen35_4b_it|multi_turn|off|vendor_preset@v1" and ident["reply_cap_tokens"] == 4096
        assert ident["attacker_model_id"] == "qwen36_27b_abliterated_attacker" and ident["context_window_tokens"] == 24576
    assert len({i["run_id"] for i in ar.values()}) == 3                  # three seeds on one engine are three runs
    mis = RP.load_plan_file(plans_dir / RP.plan_relpath("misalign_murder", "none", 2)).identities["*"]
    mis_rule, mis_detail = RP.seed_rule_of("per_response", inputs.seed_vocab)
    assert (mis["decode_seed_rule"], mis["extra"]["seed_rule_detail"]) == (mis_rule, mis_detail) == ("per_response", None)   # #212: direct
    assert mis["decode_policy_id"] == "qwen35_4b_it|agentic|off|vendor@v1"
    assert mis["token_cap_tokens"] == 3000 and mis["token_cap_ladder_json"] == [3000, 16384] and mis["n_items_planned"] == 49


def test_per_task_maps_use_the_store_writers_task_names(plans_dir):
    from sisteer.capeval.tiers.judged import TASK_ORBENCH, TASK_XSTEST
    from sisteer.capeval.tiers.parity import PARITY_TASK_MMLU
    from sisteer.capeval.tiers.petri_runner import PETRI_TASK_PREFIX
    ctl = json.loads((plans_dir / RP.store_relpath("controls", "constv2ctl2-full__lora_r4_ce", 1)).read_text())
    assert set(ctl) == {TASK_XSTEST, TASK_ORBENCH, PARITY_TASK_MMLU}
    assert ctl[TASK_XSTEST]["eval_id"] == "xstest" and ctl[TASK_ORBENCH]["eval_id"] == "or_bench_hard" and ctl[PARITY_TASK_MMLU]["eval_id"] == "mmlu"
    assert ctl[PARITY_TASK_MMLU]["decode_seed_rule"] == "fixed" and ctl[PARITY_TASK_MMLU]["decode_policy_id"] == "qwen35_4b_it|mmlu|off|greedy@v1"
    assert ctl[TASK_ORBENCH]["item_set_id"] == "or_bench_hard_200" and ctl[TASK_XSTEST]["item_set_id"] == "xstest_safe_250"
    # the map's routes are byte-identical to the per-leg files (one identity, two doors)
    for leg, task in (("xstest", TASK_XSTEST), ("or_bench", TASK_ORBENCH), ("mmlu", PARITY_TASK_MMLU)):
        assert RP.load_plan_file(plans_dir / RP.plan_relpath(leg, "constv2ctl2-full__lora_r4_ce", 1)).identities["*"] == ctl[task]
    petri = RP.load_plan_file(plans_dir / RP.plan_relpath("petri", "none", 2))
    assert set(petri.identities) == {PETRI_TASK_PREFIX + b for b in RP.PETRI_BEHAVIOURS}
    assert {i["eval_id"] for i in petri.identities.values()} == {f"petri_{b}" for b in RP.PETRI_BEHAVIOURS}
    assert len({i["run_id"] for i in petri.identities.values()}) == 6
    # the identity a container reads through the env door is exactly the map
    from sisteer.registry import hooks
    assert hooks.normalise_identities(json.loads(petri.path.read_text())) == petri.identities
    assert petri.leg == "petri" and petri.view == "s2" and petri.s == 2 and petri.served_key == "none"


def test_check_is_byte_stable_and_refuses_drift(plan_set, plans_dir, tmp_path):
    assert RP.check_plan_set(plan_set, plans_dir) == []
    other = tmp_path / "copy"
    RP.write_plan_set(plan_set, other)
    assert RP.check_plan_set(plan_set, other) == []
    p = other / RP.plan_relpath("bon", "none", 1)
    data = json.loads(p.read_text())
    data["seed_base"] = 2
    p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    (other / "bon" / "stray__s9.json").write_text("{}\n")
    problems = RP.check_plan_set(plan_set, other)
    assert f"differs: {RP.plan_relpath('bon', 'none', 1)}" in problems and "unexpected: bon/stray__s9.json" in problems


def test_refuses_an_unregistered_or_superseded_decode_row_and_a_versionless_ref(tmp_path):
    # KNOWN-BAD 1: an export that lacks the BoN vendor row -> refused, naming the exact row id. (The base export lacked it; main
    # cc994f18 / #216 landed the three vendor rows, so the absence is the fixture's `drop="bon"` — the committed export now BUILDS the
    # bon leg from its real row, checked right after.)
    inp_no_bon = RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=fixture_policy(tmp_path, drop="bon"))
    with pytest.raises(RP.RegistryPlanRefused, match=r"qwen35_4b_it\|bon\|off\|vendor_preset@v1.*not registered"):
        RP.build(inp_no_bon, ["bon"])
    # …while a leg whose row IS registered builds from the very same export (the refusal is the row's absence, not the file)
    inp_real = RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=REAL_POLICY)
    ps = RP.build(inp_real, ["pasttense"])
    assert ps.index["n_run_ids"] == 27
    assert RP.build(inp_real, ["bon"]).index["n_run_ids"] == 27                     # main #216: the real BoN vendor row resolves by exact ref
    # KNOWN-BAD 2: the multi_turn row missing alone -> the multi-turn legs refuse, named
    inp_no_mt = RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=fixture_policy(tmp_path, drop="multi_turn"))
    with pytest.raises(RP.RegistryPlanRefused, match=r"multi_turn\|off\|vendor_preset@v1.*not registered"):
        RP.build(inp_no_mt, ["crescendo"])
    # KNOWN-BAD 3: a superseded row fails closed
    inp_sup = RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=fixture_policy(tmp_path, superseded=True))
    with pytest.raises(RP.RegistryPlanRefused, match="superseded"):
        RP.build(inp_sup, ["bon"])
    # KNOWN-BAD 4: a versionless ref is not an exact row and is refused before any lookup
    with pytest.raises(RP.RegistryPlanRefused, match="carries no version"):
        RP.resolve_policy(inp_real.policy, "qwen35_4b_it|static|off|vendor_preset")
    # the known-WRONG lookup: resolve(mode=) ignores the eval and returns the IFEval row — pinned as never consulted
    wrong = inp_real.policy.resolve("qwen35_4b_it", eval_id="strongreject_bon", mode="vendor_preset")
    assert wrong.policy_id == "qwen35_4b_it|ifeval|off|vendor_preset@v1"
    src = (REPO / "scripts" / "sdfbase" / "seedrep" / "registry_plans.py").read_text()
    assert "pf.by_ref(" in src
    for forbidden in ("pf.resolve(", "policy.resolve(", "pf.find(", "policy.find(", "model_independent(", "protocol_row(", "server_default_rows("):
        assert forbidden not in src, forbidden          # the plan never consults a default / mode / find lookup on the policy file


def test_refuses_a_pin_mismatch_and_a_bad_campaign(tmp_path, inputs):
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text())
    pins["qwen35_4b"]["base64s1__constv2-mix-full__psr_ce_all"]["object_sha256"] = "0" * 63 + "x"
    bad = tmp_path / "pins.json"
    bad.write_text(json.dumps(pins))
    with pytest.raises(RP.RegistryPlanRefused, match="object_sha256"):
        RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=fixture_policy(tmp_path), pins=bad) and RP.arms_for_seed(
            RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=fixture_policy(tmp_path), pins=bad), 1)
    with pytest.raises(RP.RegistryPlanRefused, match="campaign"):
        RP.load_inputs(campaign="a:b", w_sha=W, policy=fixture_policy(tmp_path))
    with pytest.raises(RP.RegistryPlanRefused, match="--w-sha"):
        RP.load_inputs(campaign=CAMPAIGN, w_sha="0c19251", policy=fixture_policy(tmp_path))
    with pytest.raises(RP.RegistryPlanRefused, match="unknown leg"):
        RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=fixture_policy(tmp_path), windows={"nope": 1})
    # a controls leg cannot be planned without its store siblings (a partial map would misroute rows)
    with pytest.raises(RP.RegistryPlanRefused, match="store 'controls' needs all of"):
        RP.build(inputs, ["xstest"])


def test_windows_override_changes_the_instrument_identity_only_where_it_should(tmp_path):
    pol = fixture_policy(tmp_path)
    a = RP.build(RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=pol), ["crescendo", "pair"])
    b = RP.build(RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=pol, windows={"crescendo": 16384}), ["crescendo", "pair"])
    ia = a.files[RP.plan_relpath("crescendo", "none", 1)]
    ib = b.files[RP.plan_relpath("crescendo", "none", 1)]
    assert ia["context_window_tokens"] == 49152 and ib["context_window_tokens"] == 16384
    assert ia["instrument_id"] != ib["instrument_id"] and ia["run_id"] != ib["run_id"]      # a different window is a different instrument
    assert a.files[RP.plan_relpath("pair", "none", 1)] == b.files[RP.plan_relpath("pair", "none", 1)]


def test_child_env_and_queued_modal_for_plan(plans_dir, tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    plan = plans_dir / RP.plan_relpath("petri", "prompted__safety_refusal", 3)
    env = RP.child_env(plan)
    assert env == {RP.RUN_IDENTITY_ENV: str(plan.resolve())}
    spool = tmp_path / "spool"
    paths = RP.queued_modal_for_plan(plan, modal_app_id="seedrep-served-test", modal_call_id="fc-test-0001", script="scripts/modal/seedrep_served.py",
                                     spool=str(spool), registry_emit=True)
    pf = RP.load_plan_file(plan)
    assert len(paths) == 3 * len(pf.identities) == 18
    recs = [(p, rec) for p in paths for _, rec, _ in R.read_record_file(p)]
    runs = [rec["payload"] for _, rec in recs if rec["record_type"] == "run"]
    assert sorted(r["run_id"] for r in runs) == sorted(pf.run_ids.values())
    for r in runs:
        assert r["lifecycle_state"] == "queued" and r["substrate"] == "modal" and r["seed_base"] == 3 and r["extra"]["seed_rule"] == "literal_training_seed"
        assert r["decode_policy_id"] == "qwen35_4b_it|petri|off|vendor_preset@v1" and r["object_id"].startswith("si:")
    patches = [rec["payload"] for _, rec in recs if rec["record_type"] == "run_patch"]
    assert all(p["fields"] == {"modal_app_id": "seedrep-served-test", "modal_call_id": "fc-test-0001"} for p in patches)
    for p, rec in recs:
        assert isinstance(VAL.validate_record(rec, p), VAL.Validated), rec
    # the store hook reads the same file through the env door and joins THESE run ids (not ids minted from its own key)
    from sisteer.registry import hooks
    monkeypatch.setenv(RP.RUN_IDENTITY_ENV, str(plan))
    idents = hooks.normalise_identities(hooks.identity_from_env())
    assert {i["run_id"] for i in idents.values()} == set(pf.run_ids.values())
    # KNOWN-BAD: a plan file whose route lacks its run id is refused (the store would mint a different run)
    broken = tmp_path / "broken.json"
    data = json.loads(plan.read_text())
    next(iter(data.values())).pop("run_id")
    broken.write_text(json.dumps(data))
    with pytest.raises(RP.RegistryPlanRefused, match="run_id"):
        RP.load_plan_file(broken)


def test_cli_write_then_check(tmp_path):
    pol = fixture_policy(tmp_path)
    out = tmp_path / "plans"
    rc = RP.main(["--out", str(out), "--w-sha", W, "--campaign", CAMPAIGN, "--policy", str(pol), "--legs", "gsm8k", "bon"])
    assert rc == 0 and (out / RP.INDEX_NAME).is_file() and len(list(out.glob("gsm8k/*.json"))) == 27
    assert RP.main(["--out", str(out), "--w-sha", W, "--campaign", CAMPAIGN, "--policy", str(pol), "--legs", "gsm8k", "bon", "--check"]) == 0
    # the committed export carries the same BoN row since main cc994f18 (#216): it AGREES byte for byte with the fixture's plan set...
    assert RP.main(["--out", str(out), "--w-sha", W, "--campaign", CAMPAIGN, "--policy", str(REAL_POLICY), "--legs", "gsm8k", "bon", "--check"]) == 0
    # ...and a policy that DIFFERS (the BoN row superseded) refuses under --check, writing nothing
    sup = fixture_policy(tmp_path / "sup", superseded=True)
    assert RP.main(["--out", str(out), "--w-sha", W, "--campaign", CAMPAIGN, "--policy", str(sup), "--legs", "gsm8k", "bon", "--check"]) == 2
    assert RP.main(["--out", str(out), "--w-sha", W, "--campaign", CAMPAIGN, "--policy", str(pol), "--legs", "gsm8k", "bon", "--check"]) == 0   # untouched
    idx = json.loads((out / RP.INDEX_NAME).read_text())
    assert idx["legs"]["gsm8k"]["arms"]["none"]["1"]["run_ids"]["*"].startswith(f"{CAMPAIGN}:gsm8k:")
    assert idx["legs"]["gsm8k"]["arms"]["arditi_allpos"]["2"]["view"] == "arditi"


def test_seed_rule_labels_follow_the_vocabulary_present_at_run_time_and_an_unmapped_label_refuses(tmp_path, monkeypatch):
    """The ``registry-seed-rules`` contract: the writers' labels (Petri ``per_scenario_turn`` =
    ``manifest.PETRI_TARGET_SEED_RULE``, misalign ``per_response``) are emitted DIRECTLY when the tree's vocabulary spells them,
    else as ``per_item`` + ``extra.seed_rule_detail`` — decided from the vocabulary in force, never assumed; an unmapped label
    refuses; a row that names a different lane label refuses."""
    import dataclasses
    from sisteer.registry import vocab as V

    def set_vocab(vocab):
        """The tree's vocabulary as the validator sees it: the mapping AND the column plans it froze from it at import."""
        monkeypatch.setitem(V.VOCABULARIES, "DECODE_SEED_RULES", tuple(vocab))
        for key, plan in list(VAL._COLUMN_PLAN.items()):
            if plan.vocab == "DECODE_SEED_RULES":
                monkeypatch.setitem(VAL._COLUMN_PLAN, key, dataclasses.replace(plan, allowed=frozenset(vocab)))
    # STATE A (the vocabulary BEFORE main 934750dd / #212, set explicitly — this tree carries the labels since that landing, so the
    # state is set through set_vocab, never read off the tree): no per_scenario_turn / per_response / per_goal_turn -> per_item +
    # extra.seed_rule_detail. The fixture's rows spell per_item / per_draw (the export's #212 labels are replaced, see fixture_policy).
    pre_212 = tuple(v for v in V.VOCABULARIES["DECODE_SEED_RULES"] if v not in ("per_scenario_turn", "per_response", "per_goal_turn"))
    assert "per_item" in pre_212 and "per_draw" in pre_212 and len(pre_212) < len(V.VOCABULARIES["DECODE_SEED_RULES"])
    # the vocabulary in force is load_inputs' own knob (`seed_vocab=`); the record validator keeps THIS tree's vocabulary, under which
    # main's seed document (#212 labels) validates — the emitted per_item / per_draw values validate under either
    inp_a = RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=fixture_policy(tmp_path / "a", agentic_seed_rule="none"), seed_vocab=pre_212)
    assert "per_scenario_turn" not in inp_a.seed_vocab and inp_a.seed_vocab == pre_212 != tuple(V.VOCABULARIES["DECODE_SEED_RULES"])
    ps_a = RP.build(inp_a, ["petri", "misalign_murder", "pair"])
    petri_a = next(iter(ps_a.files[RP.plan_relpath("petri", "none", 1)].values()))
    assert petri_a["decode_seed_rule"] == "per_item" and petri_a["extra"]["seed_rule_detail"] == "per_scenario_turn" == RP.PETRI_ROW_SEED_RULE
    assert petri_a["extra"]["seed_rule_label"] == "per_scenario_turn" and petri_a["extra"]["seed_rule_row"] == "per_item"
    mis_a = ps_a.files[RP.plan_relpath("misalign_murder", "none", 1)]
    assert mis_a["decode_seed_rule"] == "per_item" and mis_a["extra"]["seed_rule_detail"] == "per_response" and mis_a["extra"]["seed_rule_row"] == "none"
    inst_a = ps_a.files[f"{RP.INSTRUMENTS_DIR}/{petri_a['instrument_id']}.json"]
    assert inst_a["fields"]["decode_seed_rule"] == "per_item" and inst_a["provenance"]["seed_rule_detail"] == "per_scenario_turn"
    assert inst_a["provenance"]["seed_rule_label"] == "per_scenario_turn" and inst_a["provenance"]["seed_vocabulary"] == list(inp_a.seed_vocab)
    pair_a = ps_a.files[RP.plan_relpath("pair", "none", 1)]
    assert pair_a["decode_seed_rule"] == "per_item" and pair_a["extra"]["seed_rule_detail"] == "per_goal_turn" == RP.MULTITURN_ROW_SEED_RULE
    # STATE B (the ``registry-seed-rules`` contract): the REAL export and the REAL vocabulary of main at TWO landings — 72faab6c
    # (where the labels land) and bebe99a0 (the landing the battery's W is cut after) — the labels are present, the three
    # target rows carry them -> emitted DIRECTLY, detail None. Read with `git show` at test time; skipped (never faked) when an
    # object is not in this clone. The export is byte-identical at the two shas, so the identities they mint must be identical.
    import ast
    import re
    import subprocess

    def main_bytes(sha: str, rel: str) -> bytes:
        out = subprocess.run(["git", "show", f"{sha}:{rel}"], cwd=str(REPO), capture_output=True)
        if out.returncode != 0:
            pytest.skip(f"main {sha}:{rel} is not in this clone (fetch origin/main); the labels-present state runs against the true bytes only")
        return out.stdout

    def state_b(sha: str):
        vocab_b = ast.literal_eval(re.search(r"^DECODE_SEED_RULES\s*=\s*(\([^)]*\))", main_bytes(sha, "src/sisteer/registry/vocab.py").decode("utf-8"), re.M).group(1))
        assert {"per_scenario_turn", "per_response", "per_goal_turn"} <= set(vocab_b) and "per_turn" not in vocab_b
        set_vocab(vocab_b)                                                      # what the record validator reads on that tree
        pol_b = tmp_path / "b" / f"decode_policy_main_{sha}.json"
        pol_b.parent.mkdir(parents=True, exist_ok=True)
        pol_b.write_bytes(main_bytes(sha, "configs/registry/decode_policy.json"))
        inp_b = RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=pol_b)
        assert inp_b.seed_vocab == tuple(vocab_b)
        ps_b = RP.build(inp_b, ["petri", "misalign_murder", "pair", "bon", "pasttense"])
        petri_b = next(iter(ps_b.files[RP.plan_relpath("petri", "none", 1)].values()))
        assert petri_b["decode_seed_rule"] == "per_scenario_turn" and petri_b["extra"]["seed_rule_detail"] is None
        assert petri_b["extra"]["seed_rule_row"] == "per_scenario_turn"                                   # the REAL petri row names the closure
        mis_b = ps_b.files[RP.plan_relpath("misalign_murder", "none", 1)]
        assert mis_b["decode_seed_rule"] == "per_response" and mis_b["extra"]["seed_rule_detail"] is None and mis_b["extra"]["seed_rule_row"] == "per_response"
        pair_b = ps_b.files[RP.plan_relpath("pair", "none", 1)]
        assert pair_b["decode_seed_rule"] == "per_goal_turn" and pair_b["extra"]["seed_rule_detail"] is None and pair_b["extra"]["seed_rule_row"] == "per_goal_turn"
        assert ps_b.files[RP.plan_relpath("bon", "none", 1)]["decode_seed_rule"] == "per_draw" == ps_b.files[RP.plan_relpath("bon", "none", 1)]["extra"]["seed_rule_row"]
        assert ps_b.files[RP.plan_relpath("pasttense", "none", 1)]["decode_seed_rule"] == "per_item"
        # the emitted spelling is an instrument-identity field: state B mints other instrument (and run) ids for Petri and the multi-turn legs
        assert petri_b["instrument_id"] != petri_a["instrument_id"] and petri_b["run_id"] != petri_a["run_id"] and pair_b["instrument_id"] != pair_a["instrument_id"]
        assert json.loads(RP._canonical_bytes(ps_b.files[f"{RP.INSTRUMENTS_DIR}/{petri_b['instrument_id']}.json"]))["fields"]["decode_seed_rule"] == "per_scenario_turn"
        return inp_b, ps_b, petri_b, mis_b, pair_b

    MAIN_SHAS = ("72faab6c", "bebe99a0")
    by_sha = {sha: state_b(sha) for sha in MAIN_SHAS}
    inp_b, ps_b, petri_b, mis_b, pair_b = by_sha[MAIN_SHAS[-1]]                 # bebe99a0: the state the battery's W inherits
    first = by_sha[MAIN_SHAS[0]]
    assert (petri_b["run_id"], petri_b["instrument_id"], mis_b["run_id"], mis_b["instrument_id"], pair_b["run_id"], pair_b["instrument_id"]) == \
           (first[2]["run_id"], first[2]["instrument_id"], first[3]["run_id"], first[3]["instrument_id"], first[4]["run_id"], first[4]["instrument_id"])
    assert {RP._canonical_bytes(v) for v in ps_b.files.values()} == {RP._canonical_bytes(v) for v in first[1].files.values()}   # byte-identical plan sets
    # KNOWN-BAD 1: a row carrying a LANE label other than the leg's writer label is a contradiction, refused
    pol_bad = fixture_policy(tmp_path / "bad", petri_seed_rule="per_response")
    with pytest.raises(RP.RegistryPlanRefused, match="disagree on the seed rule"):
        RP.build(RP.load_inputs(campaign=CAMPAIGN, w_sha=W, policy=pol_bad), ["petri"])
    # KNOWN-BAD 2: an unmapped writer label refuses in BOTH states (never a guessed rule)
    monkeypatch.setitem(RP.LEGS, "petri", dataclasses.replace(RP.LEGS["petri"], seed_rule="per_bogus"))
    with pytest.raises(RP.RegistryPlanRefused, match="per_bogus.*never files a guessed rule"):
        RP.build(inp_b, ["petri"])
    set_vocab(tuple(V.DECODE_SEED_RULES))
    with pytest.raises(RP.RegistryPlanRefused, match="per_bogus"):
        RP.build(inp_a, ["petri"])
    # the label is the manifest writer's symbol whenever the tree carries it (the merge check)
    from sisteer.capeval import manifest as cm
    if hasattr(cm, "PETRI_TARGET_SEED_RULE"):
        assert RP.PETRI_ROW_SEED_RULE == cm.PETRI_TARGET_SEED_RULE


def test_every_identity_carries_the_container_axis_and_the_controls_legs_share_one(plan_set, plans_dir, tmp_path):
    """``extra.container`` — the leg for a single-leg container, ``controls`` for xstest / or_bench / mmlu — on every
    identity, in the INDEX, and readable off a plan file (the completion's join key; the accepted map's third slot)."""
    idx = plan_set.index
    assert idx["containers"]["controls"] == {"legs": ["xstest", "or_bench", "mmlu"]} and len(idx["containers"]) == 13
    assert set(idx["containers"]) == {RP.container_of_leg(l) for l in RP.LEG_NAMES}
    for pf in RP.iter_plan_files(plans_dir):
        want = "controls" if pf.leg in ("xstest", "or_bench", "mmlu") else pf.leg
        assert pf.container == want == RP.container_of_leg(pf.leg)
        assert all(ident["extra"]["container"] == want for ident in pf.identities.values())
    ctl = RP.load_plan_file(plans_dir / RP.store_relpath("controls", "none", 2))
    assert ctl.container == "controls" and ctl.leg == "mmlu+or_bench+xstest" and ctl.view == "s2"
    # KNOWN-BAD: an identity with no extra.container is REFUSED at load — a container is never guessed from the leg
    ident = dict(RP.load_plan_file(plans_dir / RP.plan_relpath("bon", "none", 1)).identities["*"])
    ident["extra"] = {k: v for k, v in ident["extra"].items() if k != "container"}
    stray = tmp_path / "no_container.json"
    stray.write_text(json.dumps(ident))
    with pytest.raises(RP.RegistryPlanRefused, match="lacks extra.container"):
        RP.load_plan_file(stray)
