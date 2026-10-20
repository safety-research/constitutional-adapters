"""The job-side plan builder (``sisteer.registry.jobplan``): from a resolved config + the REAL registry seeds
(``configs/registry``) + a roster / pins pair, one validated plan per (declared eval x declared object) —
object ids from the pins (a staged object), the SI texts (a prompted entry) and ``none``; the canonical
seeded instrument per eval for the served model and path (the SAME ``instrument_id`` the dimensions loader
mints); the eval's producer spellings as routing facts. Anything it cannot state honestly is skipped by name;
an unknown model key or an empty eval list refuses the whole registration. Module import is stdlib-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from sisteer.registry import emit, identity, jobplan as JP
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import reconcile as RC
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import dimensions as DIM
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
SEEDS = REPO / "configs" / "registry"
STAGED_SHA = hashlib.sha256(b"synthetic staged object").hexdigest()
CONST_SHA = "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"   # si_texts.yaml si_constitution


@pytest.fixture(scope="module")
def seed_index():
    return JP.load_seed_index(str(SEEDS))


def synthetic_checkout(tmp_path: Path, objects=("none", "synth__lora_r4", "prompted__si_constitution", "unmapped__x")) -> dict:
    """A fake checkout: the real seeds (symlinked), a synthetic config / roster / pins under ``configs/``."""
    root = tmp_path / "checkout"
    (root / "configs").mkdir(parents=True)
    os.symlink(SEEDS, root / "configs" / "registry")
    cfg = root / "configs" / "synth_full22_qwen35_4b.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": list(objects), "model": "Qwen/Qwen3.5-4B"}))
    roster = root / "configs" / "sdfbase_full22_roster.yaml"
    roster.write_text(yaml.safe_dump({"targets": {"qwen35_4b": {"views": {
        "main": {"cells": {"synth__lora_r4": "base__synth__lora_r4", "prompted__si_constitution": None}},
        "it": {"cells": {"synth__lora_r4": "itdoc__synth__lora_r4"}}}}}}))
    pins = root / "configs" / "sdfbase_full22_pins.json"
    pins.write_text(json.dumps({"qwen35_4b": {"base__synth__lora_r4": {"object_sha256": STAGED_SHA},
                                              "itdoc__synth__lora_r4": {"object_sha256": STAGED_SHA[::-1]}}}))
    return {"root": root, "cfg": cfg, "roster": roster, "pins": pins}


def test_module_imports_under_python_dash_S():
    out = subprocess.run(["python3", "-S", "-c", "import sisteer.registry.jobplan, sys; "
                          "print(sorted(m for m in sys.modules if m.split('.')[0] in ('yaml','torch','pandas','numpy')))"],
                         capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": os.environ.get("PATH", "")}, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_build_plans_declares_one_run_per_eval_x_object(tmp_path, seed_index):
    co = synthetic_checkout(tmp_path)
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["petri_self_preservation", "agentic_murder", "nonsuch"],
                       serving_path="lora_native", campaign="sdfbase_full22", job_id="4242", job_name="f22-petri-qwen35_4b",
                       substrate="cluster_a", code_sha=synth.code_sha("a"))
    # 2 resolvable evals x 3 resolvable objects; the unmapped object and the unknown eval are skipped BY NAME
    assert len(b.runs) == 6
    skipped = dict(b.skipped)
    assert "object unmapped__x" in skipped and "roster maps no staged id" in skipped["object unmapped__x"]
    assert "eval nonsuch" in skipped and "no canonical seeded instrument" in skipped["eval nonsuch"]
    by = {(r.route["eval_id"], r.route["object"]): r for r in b.runs}
    assert by[("petri_self_preservation", "none")].plan["object_id"] == "none"
    assert by[("petri_self_preservation", "synth__lora_r4")].plan["object_id"] == STAGED_SHA
    assert by[("petri_self_preservation", "prompted__si_constitution")].plan["object_id"] == f"si:{CONST_SHA}"
    assert len({r.run_id for r in b.runs}) == 6
    # the plan is the registry's run identity, validated as the record it becomes
    producer = emit.producer_block("t", code_sha=synth.code_sha("a"), substrate="cluster_a")
    for r in b.runs:
        p = r.plan
        assert p["model_id"] == "qwen35_4b_it" and p["campaign"] == "sdfbase_full22"
        assert p["producer_run_key_legacy"] == "f22-petri-qwen35_4b@4242" and p["producer_object_key_legacy"] == r.route["object"]
        assert p["lifecycle_state"] == "running" and p["started_utc"] == p["queued_utc"]
        assert p["extra"]["registered_at"] == "job_start" and p["extra"]["evals_declared"] == ["petri_self_preservation", "agentic_murder", "nonsuch"]
        assert p["cluster_job_id"] == "4242" and p["job_name"] == "f22-petri-qwen35_4b"
        assert p["transfer_kind"] == ("none" if p["object_id"] == "none" else "base_to_instruct")
        assert p["applied_positions_mode"] == ("none" if p["object_id"] == "none" else "all")
        assert emit.check_planned_run(p, producer) == r.run_id
        ids, run_payload, state_payload = emit.planned_run_records(p, producer)
        for rt, payload in (("run", run_payload), ("run_state", state_payload)):
            rec = R.make_record(rt, payload, producer, identity.producer_record_id(), R.utc_now())
            assert isinstance(VAL.validate_record(rec, "<plan>"), VAL.Validated), (rt, VAL.validate_record(rec, "<plan>"))
    # the instrument is the seeds' canonical one for the eval — the id the dimensions loader mints
    seeds = DIM.load_seeds(str(SEEDS))
    bundle = DIM.build_records(seeds, emit.producer_block("t"), verify_si=False)
    petri_ids = bundle.instrument_ids["petri_full22_as_run"]
    assert by[("petri_self_preservation", "none")].plan["instrument_id"] in petri_ids
    agentic = by[("agentic_murder", "none")].plan
    assert agentic["instrument_id"] in bundle.instrument_ids["agentic_misalignment_runner_T1_3000_16384"]
    assert agentic["item_set_id"] == "agentic_murder_scenarios" and agentic["harness"] == "agentic_misalignment"
    assert agentic["token_cap_tokens"] == 16384 and agentic["scoring_rule"] == "rate_over_scenarios"
    # routing facts: every producer spelling of the eval, the eval id first
    assert by[("agentic_murder", "none")].route["tasks"][0] == "agentic_murder" and "murder" in by[("agentic_murder", "none")].route["tasks"]
    assert "self_preservation" in by[("petri_self_preservation", "none")].route["tasks"]


def test_a_planned_run_states_no_ladder_index_and_the_cap_rule_reaches_one_that_does(tmp_path, seed_index):
    """The job-side plan carries the instrument's cap and no ladder index -- the legacy shape `registry cap-check`
    counts, admitted by the record rule -- so every planned run still validates; a plan that states an index must agree with its ladder and
    cap (the rule reaches the planned record like any `run` record)."""
    co = synthetic_checkout(tmp_path)
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["agentic_murder"], serving_path="lora_native",
                       campaign="sdfbase_full22", job_id="4243", job_name="f22-agentic-qwen35_4b", substrate="cluster_a", code_sha=synth.code_sha("a"))
    assert b.runs
    producer = emit.producer_block("t", code_sha=synth.code_sha("a"), substrate="cluster_a")
    for r in b.runs:
        _ids, run_payload, _state = emit.planned_run_records(r.plan, producer)
        assert run_payload["token_cap_tokens"] == 16384 and run_payload.get("token_cap_rung_index") is None

        def as_record(payload):
            return R.make_record("run", payload, producer, identity.producer_record_id(), R.utc_now())

        assert isinstance(VAL.validate_record(as_record(run_payload), "<plan>"), VAL.Validated)
        stated = dict(run_payload, token_cap_rung_index=1, token_cap_ladder_json=[3000, 16384])
        assert isinstance(VAL.validate_record(as_record(stated), "<plan>"), VAL.Validated)
        wrong = dict(stated, token_cap_rung_index=0)
        out = VAL.validate_record(as_record(wrong), "<plan>")
        assert isinstance(out, VAL.Malformed) and out.field == "token_cap_tokens" and "D648" in out.reason


def test_a_per_behaviour_petri_run_plans_against_its_own_item_set_when_the_seeds_carry_it(tmp_path, seed_index, monkeypatch):
    """With the six per-behaviour sets in the seed index, `petri_self_preservation` plans item_set_id
    petri_scenarios_self_preservation and n_items_planned 10 while `petri_pooled` keeps petri_scenarios_6x10 / 60 and the instrument id
    is unchanged; a seed index WITHOUT the set (today's stage-e-run seeds) registers exactly as before (6x10 / 60)."""
    import copy
    co = synthetic_checkout(tmp_path, objects=("none",))
    kw = dict(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["petri_self_preservation", "petri_pooled"], serving_path="lora_native",
              campaign="sdfbase_full22", job_id="4242", job_name="f22-petri-qwen35_4b", substrate="cluster_a", code_sha=synth.code_sha("a"))
    before = {r.route["eval_id"]: r.plan for r in JP.build_plans(**kw).runs}
    pooled_set = before["petri_pooled"]["item_set_id"]
    assert pooled_set == "petri_scenarios_6x10" and before["petri_pooled"]["n_items_planned"] == 60
    if "petri_scenarios_self_preservation" not in seed_index.item_sets:        # the seeds before (a) folds: unchanged behaviour, pinned
        assert before["petri_self_preservation"]["item_set_id"] == pooled_set and before["petri_self_preservation"]["n_items_planned"] == 60
    # the seed index WITH the partition (what (a) lands): the behaviour run follows its own set, the pooled run and the instrument do not move
    idx = copy.deepcopy(seed_index)
    for b in ("self_preservation", "delusion_sycophancy"):
        idx.item_sets[f"petri_scenarios_{b}"] = {"item_set_id": f"petri_scenarios_{b}", "eval_id": f"petri_{b}", "n_items": 10,
                                                 "dataset_version": "frozen scenario bundle 8812ba083bb5473c", "items_sha256": "unverified",
                                                 "source_uri": "test", "item_id_scheme": "<behaviour>/<scenario>", "subsample_rule": "fixed_ids_file"}
    monkeypatch.setattr(JP, "load_seed_index", lambda _dir: idx)
    after = {r.route["eval_id"]: r.plan for r in JP.build_plans(**kw).runs}
    assert after["petri_self_preservation"]["item_set_id"] == "petri_scenarios_self_preservation" and after["petri_self_preservation"]["n_items_planned"] == 10
    assert after["petri_pooled"]["item_set_id"] == pooled_set and after["petri_pooled"]["n_items_planned"] == 60
    assert after["petri_self_preservation"]["instrument_id"] == before["petri_self_preservation"]["instrument_id"]
    assert after["petri_self_preservation"]["eval_id"] == "petri_self_preservation"
    # the plan still validates as the run record it becomes
    producer = emit.producer_block("t", code_sha=synth.code_sha("a"), substrate="cluster_a")
    for p in after.values():
        _ids, run_payload, state_payload = emit.planned_run_records(p, producer)
        rec = R.make_record("run", run_payload, producer, identity.producer_record_id(), R.utc_now())
        assert isinstance(VAL.validate_record(rec, "<plan>"), VAL.Validated), VAL.validate_record(rec, "<plan>")
    # the helper alone: a set declared for ANOTHER eval is never borrowed (spelling is not ownership); a non-Petri eval is untouched
    inst = {"item_set_id": pooled_set, "eval_id": "petri_self_preservation"}
    idx.item_sets["petri_scenarios_defer_to_users"] = {"item_set_id": "petri_scenarios_defer_to_users", "eval_id": "petri_pooled", "n_items": 10}
    assert JP.per_behaviour_item_set(idx, "petri_defer_to_users", {"item_set_id": pooled_set})[0]["item_set_id"] == pooled_set
    assert JP.per_behaviour_item_set(idx, "agentic_murder", {"item_set_id": "agentic_murder_scenarios"})[0]["item_set_id"] == "agentic_murder_scenarios"
    assert JP.per_behaviour_item_set(idx, "petri_pooled", inst)[0]["item_set_id"] == pooled_set


def test_the_per_behaviour_item_set_is_selected_by_eval_ownership_not_spelling(seed_index):
    """A renamed / versioned ten-scenario set that still declares `eval_id: petri_<behaviour>` is
    found (no silent fall back to the pooled sixty); two sets claiming one eval refuse by name; a claimant without a usable n_items
    refuses; no claimant at all keeps the instrument's own set (the earlier seeds). The spelling rule is gone."""
    import copy
    idx = copy.deepcopy(seed_index)
    idx.item_sets.pop("petri_scenarios_self_preservation", None)       # the real per-behaviour set (seeds since 7225cde85): the test declares its own owners
    inst = {"item_set_id": "petri_scenarios_6x10", "eval_id": "petri_self_preservation"}
    idx.item_sets["petri_scenarios_self_preservation_v2"] = {"item_set_id": "petri_scenarios_self_preservation_v2", "eval_id": "petri_self_preservation", "n_items": 10}
    got_inst, got_set = JP.per_behaviour_item_set(idx, "petri_self_preservation", inst)
    assert got_inst["item_set_id"] == "petri_scenarios_self_preservation_v2" and got_set["n_items"] == 10
    idx.item_sets["petri_scenarios_self_preservation"] = {"item_set_id": "petri_scenarios_self_preservation", "eval_id": "petri_self_preservation", "n_items": 10}
    with pytest.raises(JP.JobPlanRefused, match="2 item sets declare it .*petri_scenarios_self_preservation, petri_scenarios_self_preservation_v2"):
        JP.per_behaviour_item_set(idx, "petri_self_preservation", inst)
    del idx.item_sets["petri_scenarios_self_preservation"]
    for bad in (None, 0, True, "10"):
        idx.item_sets["petri_scenarios_self_preservation_v2"]["n_items"] = bad
        with pytest.raises(JP.JobPlanRefused, match="no usable n_items"):
            JP.per_behaviour_item_set(idx, "petri_self_preservation", inst)
    del idx.item_sets["petri_scenarios_self_preservation_v2"]
    assert JP.per_behaviour_item_set(idx, "petri_self_preservation", inst)[0]["item_set_id"] == "petri_scenarios_6x10"
    assert not hasattr(JP, "PETRI_BEHAVIOUR_ITEM_SET_PREFIX")
    # The instrument's OWN set counts as an owner. Own set A declaring the eval + a versioned replacement B
    # declaring it too → refused by name (never instrument A with set B); own set A as the sole owner → kept as is
    idx.item_sets["set_a"] = {"item_set_id": "set_a", "eval_id": "petri_self_preservation", "n_items": 10}
    own_inst = {"item_set_id": "set_a", "eval_id": "petri_self_preservation"}
    kept_inst, kept_set = JP.per_behaviour_item_set(idx, "petri_self_preservation", own_inst)
    assert kept_inst == own_inst and kept_set is idx.item_sets["set_a"]
    idx.item_sets["set_b"] = {"item_set_id": "set_b", "eval_id": "petri_self_preservation", "n_items": 10}
    with pytest.raises(JP.JobPlanRefused, match="2 item sets declare it .*set_a, set_b; the instrument binds set_a"):
        JP.per_behaviour_item_set(idx, "petri_self_preservation", own_inst)


def test_two_claimant_item_sets_refuse_the_whole_registration_by_name(tmp_path, seed_index, monkeypatch):
    """The ambiguity reaches the registration as a whole (JobPlanRefused), never a per-eval skip that registers the pooled sixty."""
    import copy
    idx = copy.deepcopy(seed_index)
    idx.item_sets.pop("petri_scenarios_self_preservation", None)       # the real set would be a third owner; the test's two are the point
    idx.item_sets["a_set"] = {"item_set_id": "a_set", "eval_id": "petri_self_preservation", "n_items": 10}
    idx.item_sets["b_set"] = {"item_set_id": "b_set", "eval_id": "petri_self_preservation", "n_items": 10}
    monkeypatch.setattr(JP, "load_seed_index", lambda _dir: idx)
    co = synthetic_checkout(tmp_path, objects=("none",))
    with pytest.raises(JP.JobPlanRefused, match="2 item sets declare it"):
        JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["petri_self_preservation"], serving_path="lora_native",
                       campaign="sdfbase_full22", job_id="4242", job_name="f22-petri-qwen35_4b", substrate="cluster_a", code_sha=synth.code_sha("a"))


def test_build_plans_selection_and_refusals(tmp_path):
    co = synthetic_checkout(tmp_path)
    # the `it` selection: the model key carries the suffix, the roster is read under the target key, the pins under the it staged id
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b_it", roster_selection="it", evals=["xstest"], serving_path="lora_native",
                       campaign="sdfbase_full22", job_id="1", job_name="j", substrate="cluster_a", code_sha=synth.code_sha("a"))
    objs = {r.route["object"]: r.plan for r in b.runs}
    assert objs["synth__lora_r4"]["object_id"] == STAGED_SHA[::-1] and objs["synth__lora_r4"]["transfer_kind"] == "instruct_direct"
    assert "prompted__si_constitution" in objs          # a prompted entry needs no roster row
    # whole-registration refusals: an unknown model key, no evals, a missing config
    with pytest.raises(JP.JobPlanRefused, match="not a legacy model spelling"):
        JP.build_plans(config_path=str(co["cfg"]), model_key="nonsuch_model", evals=["xstest"], serving_path="lora_native",
                       campaign="c", job_id="1", job_name="j")
    with pytest.raises(JP.JobPlanRefused, match="no evals"):
        JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=[], serving_path="lora_native", campaign="c", job_id="1", job_name="j")
    with pytest.raises(JP.JobPlanRefused, match="does not exist"):
        JP.build_plans(config_path=str(tmp_path / "nope.yaml"), model_key="qwen35_4b", evals=["xstest"], serving_path="lora_native",
                       campaign="c", job_id="1", job_name="j")
    # a serving path with no canonical instrument for the eval is a per-eval skip, not a guess
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["petri_pooled"], serving_path="hf",
                       campaign="sdfbase_full22", job_id="2", job_name="j", substrate="cluster_a", code_sha=synth.code_sha("a"))
    assert not b.runs and any("petri_pooled" in w for w, _ in b.skipped)


def test_canonical_instrument_is_unique_per_eval_for_the_served_models(seed_index):
    """Every canonical seeded eval resolves to exactly ONE instrument on the native path for a served model, or is
    refused by name — the builder never picks among several."""
    ok, refused = [], []
    for eval_id in sorted(seed_index.evals):
        try:
            JP.canonical_instrument(seed_index, eval_id, "qwen35_4b_it", "lora_native")
            ok.append(eval_id)
        except LookupError as exc:
            refused.append((eval_id, str(exc)))
    assert {"petri_pooled", "xstest", "or_bench_hard", "agentic_murder", "pasttense", "strongreject_bon", "crescendo", "gsm8k", "math500"} <= set(ok)
    for eval_id, why in refused:
        assert "no canonical seeded instrument" in why or "canonical seeded instruments" in why, (eval_id, why)


#: the two multi-turn evals whose seeds carry a `non_template` variant beside the canonical every-token row, for the 35B model alone
NTPL_EVALS = ("actorattack", "crescendo")
NTPL_MODEL = "qwen35_35b_a3b"


def test_canonical_instrument_selects_by_served_positions_mode(seed_index):
    """Mode-aware selection at registration: without a mode the canonical row (byte-identical to the rule before the mode existed);
    with `non_template` the seeded `sensitivity` variant, whose identity differs from the canonical row's in `applied_positions_mode`
    alone; with `all` the canonical row again (several candidates carry `all`? no -- but the canonical one wins when they do); with a
    mode no seed carries for the eval, or a mode outside the vocabulary, a refusal naming the eval and the mode."""
    for eval_id in NTPL_EVALS:
        canonical, canonical_extra = JP.canonical_instrument(seed_index, eval_id, NTPL_MODEL, "lora_native")
        assert canonical_extra["paper_status"] == "canonical" and canonical["applied_positions_mode"] == "all"
        # the 35B's canonical every-token row is the vendor-preset seed since the 2026-10-14 split by decode route (its vendor row requires
        # per_goal_turn, which is instrument identity); the generation_config seed keeps the paper's served models alone
        assert canonical_extra["seed_instrument_id"] == f"multiturn_{eval_id}_cap4096_vendor_preset" and canonical["decode_seed_rule"] == "per_goal_turn"
        variant, variant_extra = JP.canonical_instrument(seed_index, eval_id, NTPL_MODEL, "lora_native", positions_mode="non_template")
        assert variant_extra["seed_instrument_id"] == f"multiturn_{eval_id}_cap4096_generation_config_non_template"
        assert variant_extra["paper_status"] == "sensitivity" and variant_extra["variant_of"] == canonical_extra["seed_instrument_id"]
        assert variant["applied_positions_mode"] == "non_template" and variant["instrument_id"] != canonical["instrument_id"]
        swapped = identity.instrument_identity_fields(canonical)
        swapped["applied_positions_mode"] = "non_template"
        assert identity.instrument_id(swapped) == variant["instrument_id"], "the variant's identity is the canonical row's with the mode swapped"
        assert JP.canonical_instrument(seed_index, eval_id, NTPL_MODEL, "lora_native", positions_mode="all")[0]["instrument_id"] == canonical["instrument_id"]
        with pytest.raises(LookupError, match=f"applied_positions_mode 'response' for eval '{eval_id}'"):
            JP.canonical_instrument(seed_index, eval_id, NTPL_MODEL, "lora_native", positions_mode="response")
        with pytest.raises(LookupError, match="not an applied-positions mode"):
            JP.canonical_instrument(seed_index, eval_id, NTPL_MODEL, "lora_native", positions_mode="sideways")
        # another served model has no non-template seed: refused by name, never the 35B's row
        with pytest.raises(LookupError, match=f"applied_positions_mode 'non_template' for eval '{eval_id}' on qwen35_4b_it"):
            JP.canonical_instrument(seed_index, eval_id, "qwen35_4b_it", "lora_native", positions_mode="non_template")


def test_canonical_instrument_with_a_mode_prefers_the_canonical_row_among_several_and_refuses_otherwise():
    """The rule on a hand-built index: two seeds carrying the mode -> the canonical one; two non-canonical ones -> refused naming both."""
    def row(seed_id, status, mode, inst_id):
        return ({"eval_id": "ev", "item_set_id": "is", "serving_path": "lora_native", "applied_positions_mode": mode, "instrument_id": inst_id},
                {"seed_instrument_id": seed_id, "paper_status": status, "model_ids": ["m"]})
    idx = JP.SeedIndex([row("a", "canonical", "all", "i1"), row("b", "sensitivity", "all", "i2")], {}, {}, {}, [], "rev")
    assert JP.canonical_instrument(idx, "ev", "m", "lora_native", positions_mode="all")[0]["instrument_id"] == "i1"
    assert JP.canonical_instrument(idx, "ev", "m", "lora_native")[0]["instrument_id"] == "i1"
    idx2 = JP.SeedIndex([row("a", "sensitivity", "non_template", "i1"), row("b", "sensitivity", "non_template", "i2")], {}, {}, {}, [], "rev")
    with pytest.raises(LookupError, match=r"2 seeded instruments with applied_positions_mode 'non_template'.*\['a', 'b'\]"):
        JP.canonical_instrument(idx2, "ev", "m", "lora_native", positions_mode="non_template")
    with pytest.raises(LookupError, match="no canonical seeded instrument"):
        JP.canonical_instrument(idx2, "ev", "m", "lora_native")


def test_build_plans_with_a_positions_mode_registers_on_the_variant_and_derives_the_scope(tmp_path, seed_index):
    """`build_plans(positions_mode=...)` threads the mode: the plan's instrument_id is the variant's, its applied_positions_mode the
    variant's, `extra.positions_mode_requested` records the request, and `multi_turn_steering_scope` is DERIVED from the mode and the
    seeded eval's category (every_turn_prefill_and_decode for non_template on a multi-turn eval; never the single-turn default) with its
    derivation in field_confidence_json. Without a mode the same job registers on the canonical row with no request recorded, and a
    multi-turn eval still derives its scope."""
    co = synthetic_checkout(tmp_path, objects=("none", "synth__lora_r4"))
    # the synthetic checkout's roster is keyed qwen35_4b; the 35B seeds are reached through the model key of the 35B
    cfg = co["root"] / "configs" / "synth_full22_qwen35_35b.yaml"
    cfg.write_text(yaml.safe_dump({"explicit_cells": ["none", "synth__lora_r4"], "model": "Qwen/Qwen3.5-35B-A3B"}))
    roster = yaml.safe_load(co["roster"].read_text())
    roster["targets"]["qwen35_35b_a3b"] = roster["targets"]["qwen35_4b"]
    co["roster"].write_text(yaml.safe_dump(roster))
    pins = json.loads(co["pins"].read_text())
    pins["qwen35_35b_a3b"] = pins["qwen35_4b"]
    co["pins"].write_text(json.dumps(pins))
    common = dict(config_path=str(cfg), model_key="qwen35_35b_a3b", evals=list(NTPL_EVALS), serving_path="lora_native", campaign="sdfbase_full22",
                  job_id="31", job_name="ntpl", substrate="cluster_a", code_sha=synth.code_sha("a"))
    b = JP.build_plans(positions_mode="non_template", **common)
    assert not b.skipped, b.skipped
    assert any("positions mode requested: non_template" in n for n in b.notes)
    by = {(r.route["eval_id"], r.route["object"]): r.plan for r in b.runs}
    assert set(by) == {(e, o) for e in NTPL_EVALS for o in ("none", "synth__lora_r4")}
    for eval_id in NTPL_EVALS:
        variant, variant_extra = JP.canonical_instrument(seed_index, eval_id, NTPL_MODEL, "lora_native", positions_mode="non_template")
        steered, unsteered = by[(eval_id, "synth__lora_r4")], by[(eval_id, "none")]
        for p in (steered, unsteered):
            assert p["instrument_id"] == variant["instrument_id"] and p["extra"]["seed_instrument_id"] == variant_extra["seed_instrument_id"]
            assert p["extra"]["positions_mode_requested"] == "non_template"
        assert steered["applied_positions_mode"] == "non_template" and steered["multi_turn_steering_scope"] == "every_turn_prefill_and_decode"
        assert steered["field_confidence_json"]["multi_turn_steering_scope"]["confidence"] == "derived"
        assert "applied_positions_mode=non_template on a multi-turn eval" in steered["field_confidence_json"]["multi_turn_steering_scope"]["source"]
        assert unsteered["applied_positions_mode"] == "none" and unsteered["multi_turn_steering_scope"] == "not_applicable"
        assert "multi_turn_steering_scope" not in unsteered["field_confidence_json"]
    b0 = JP.build_plans(**common)
    assert not b0.skipped, b0.skipped
    by0 = {(r.route["eval_id"], r.route["object"]): r.plan for r in b0.runs}
    for eval_id in NTPL_EVALS:
        canonical, _ = JP.canonical_instrument(seed_index, eval_id, NTPL_MODEL, "lora_native")
        p = by0[(eval_id, "synth__lora_r4")]
        assert p["instrument_id"] == canonical["instrument_id"] and "positions_mode_requested" not in p["extra"]
        assert p["applied_positions_mode"] == "all" and p["multi_turn_steering_scope"] == "every_turn_prefill_and_decode"
    # a mode no seed carries: every eval skipped by name, nothing registered
    b2 = JP.build_plans(positions_mode="response", **common)
    assert not b2.runs and len(b2.skipped) == len(NTPL_EVALS) and all("applied_positions_mode 'response'" in why for _, why in b2.skipped)


def test_build_plans_derives_single_turn_for_a_single_turn_eval_and_reads_the_category_from_the_seeds(tmp_path):
    """The derivation on the 4B: a static eval reads single_turn (derived, not defaulted), a multi-turn one every_turn_prefill_and_decode
    under the canonical `all` row, an unsteered pair not_applicable."""
    co = synthetic_checkout(tmp_path, objects=("none", "synth__lora_r4"))
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["pasttense", "crescendo"], serving_path="lora_native",
                       campaign="sdfbase_full22", job_id="32", job_name="j", substrate="cluster_a", code_sha=synth.code_sha("a"))
    assert not b.skipped, b.skipped
    by = {(r.route["eval_id"], r.route["object"]): r.plan for r in b.runs}
    assert by[("pasttense", "synth__lora_r4")]["multi_turn_steering_scope"] == "single_turn"
    assert by[("pasttense", "synth__lora_r4")]["field_confidence_json"]["multi_turn_steering_scope"] == {
        "confidence": "derived", "source": "eval_category=static_attack is single-turn"}
    assert by[("crescendo", "synth__lora_r4")]["multi_turn_steering_scope"] == "every_turn_prefill_and_decode"
    assert by[("crescendo", "none")]["multi_turn_steering_scope"] == "not_applicable" and by[("pasttense", "none")]["multi_turn_steering_scope"] == "not_applicable"


SI_SCOPE_SOURCE = "object_kind=system_instruction: a prompted object steers no positions"


def test_a_prompted_entry_registers_not_applicable_scope_on_a_multi_turn_eval_and_its_live_row_reads_it_after_the_fold(tmp_path, monkeypatch):
    """A `prompted__<si>` entry's object is a system instruction (`si:<sha>`), which steers no positions: its plan's scope is
    not_applicable with the kind's derivation on the Petri and agentic evals -- never every_turn_prefill_and_decode, whatever mode the
    selected instrument declares (the plan still stamps the instrument's `all` on the mode column, the brief's rule) -- and single_turn on
    a static eval, where the category decides first. The staged LoRA beside it derives every_turn_prefill_and_decode from that same mode,
    the anchor not_applicable without an entry. Registered through the real seeds into a temp root: the rows read so after the fold, the
    reconcile selects none of them, and the database rebuilds equal."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    co = synthetic_checkout(tmp_path, objects=("none", "synth__lora_r4", "prompted__si_constitution"))
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["petri_self_preservation", "agentic_murder", "pasttense"],
                       serving_path="lora_native", campaign="sdfbase_full22", job_id="4343", job_name="f22-prompted-qwen35_4b",
                       substrate="cluster_a", code_sha=synth.code_sha("a"))
    assert not b.skipped, b.skipped
    by = {(r.route["eval_id"], r.route["object"]): r.plan for r in b.runs}
    assert len(by) == 9
    for eval_id in ("petri_self_preservation", "agentic_murder"):
        prompted = by[(eval_id, "prompted__si_constitution")]
        assert prompted["object_id"] == f"si:{CONST_SHA}" and identity.object_kind_from_id(prompted["object_id"]) == "system_instruction"
        assert prompted["applied_positions_mode"] == "all"                                   # the selected instrument's, as every plan's
        assert prompted["multi_turn_steering_scope"] == "not_applicable"
        assert prompted["field_confidence_json"]["multi_turn_steering_scope"] == {"confidence": "derived", "source": SI_SCOPE_SOURCE}
        staged = by[(eval_id, "synth__lora_r4")]
        assert staged["applied_positions_mode"] == "all" and staged["multi_turn_steering_scope"] == "every_turn_prefill_and_decode"
        assert "applied_positions_mode=all on a multi-turn eval" in staged["field_confidence_json"]["multi_turn_steering_scope"]["source"]
        anchor = by[(eval_id, "none")]
        assert anchor["multi_turn_steering_scope"] == "not_applicable" and "multi_turn_steering_scope" not in anchor["field_confidence_json"]
    static = by[("pasttense", "prompted__si_constitution")]
    assert static["multi_turn_steering_scope"] == "single_turn"
    assert static["field_confidence_json"]["multi_turn_steering_scope"] == {"confidence": "derived", "source": "eval_category=static_attack is single-turn"}
    # through the real seeds into a temp root (the staged object's row is synthetic, fitted on the seeds' 4B base checkpoint: the seeds
    # stage no LoRA): the planned records validate, fold without a refusal or a block, and the live rows read the plans' scopes
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    DIM.write_seed_records(root, str(SEEDS), producer=emit.producer_block("t", code_sha=synth.code_sha("a")), verify_si=False)
    staged = dict(synth.object_lora(sha=STAGED_SHA), fit_model_id="qwen35_4b_base")
    R.write_records(os.path.join(root, "records", "dims", "000001-staged.jsonl"), [synth.rec("object", staged, synth.t(1))])
    producer = emit.producer_block("t", code_sha=synth.code_sha("a"), substrate="cluster_a")
    for r in b.runs:
        _ids, run_payload, state_payload = emit.planned_run_records(r.plan, producer)
        recs = [R.make_record(rt, payload, producer, identity.producer_record_id(), synth.t(10 + i))
                for i, (rt, payload) in enumerate((("run", run_payload), ("run_state", state_payload)))]
        for rec in recs:
            assert isinstance(VAL.validate_record(rec, "<plan>"), VAL.Validated), VAL.validate_record(rec, "<plan>")
        R.write_records(os.path.join(root, "records", "sdfbase_full22", identity.run_dir_name(r.run_id), "000001-run.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    live = {rid: (kind, mode, scope) for rid, kind, mode, scope in conn.execute(
        "SELECT r.run_id, o.object_kind, r.applied_positions_mode, r.multi_turn_steering_scope FROM run r JOIN object o ON o.object_id = r.object_id")}
    conn.close()
    assert len(live) == 9
    for r in b.runs:
        kind, mode, scope = live[r.run_id]
        if r.route["object"] == "prompted__si_constitution":
            assert kind == "system_instruction" and mode == "all"
            assert scope == ("single_turn" if r.route["eval_id"] == "pasttense" else "not_applicable"), r.route
        elif r.route["object"] == "none":
            assert (kind, mode, scope) == ("none", "none", "not_applicable"), r.route
        else:
            assert kind == "lora" and mode == "all" and scope == ("single_turn" if r.route["eval_id"] == "pasttense" else "every_turn_prefill_and_decode"), r.route
    # nothing for the reconcile to correct: no row carries the default on a multi-turn eval; and the database rebuilds equal
    assert RC.reconcile_multi_turn_scope(root, apply=False).runs == []
    live_db = DB.connect(DB.db_path(root), readonly=True)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = {k: v for k, v in H.verify_equal(live_db, rebuilt).items() if v}
    assert not diff, diff
    live_db.close()
    rebuilt.close()


def test_pending_document_shape(tmp_path):
    co = synthetic_checkout(tmp_path, objects=("none",))
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b", evals=["xstest", "or_bench_hard"], serving_path="lora_native",
                       campaign="sdfbase_full22", job_id="7", job_name="f22s-qwen35_4b", substrate="cluster_a", code_sha=synth.code_sha("a"))
    doc = b.as_pending("7", "f22s-qwen35_4b")
    assert doc["run_id"] == b.runs[0].run_id and doc["plan"]["run_id"] == b.runs[0].run_id and doc["registered_at"] == "job_start"
    assert [r["run_id"] for r in doc["runs"]] == [r.run_id for r in b.runs]
    assert all(r["plan"]["run_id"] == r["run_id"] and r["route"]["object"] == "none" for r in doc["runs"])
    json.dumps(doc)     # serialisable as written


def test_seed_replicate_model_keys_register_through_the_seed_selection(tmp_path, seed_index):
    """The seed-replicate battery: the harness registers a seed selection under the model key `qwen35_4b_s<s>` (KEY + the
    roster selection's suffix, `common.sh f22_registry_register`), which `model_for_key` REFUSED as not a legacy spelling — 24
    cluster B static jobs (2026-09-20) ran unregistered. The three keys now spell the served instruct checkpoint with the
    base-checkpoint fit (the seed rides the selection / the object, never the model); a fourth seed still refuses by name (the
    table is exact-match, one cited row per spelling)."""
    for s in (1, 2, 3):
        assert JP.model_for_key(seed_index.legacy_model_keys, f"qwen35_4b_s{s}") == ("qwen35_4b_it", "base_to_instruct")
    with pytest.raises(JP.JobPlanRefused, match="not a legacy model spelling"):
        JP.model_for_key(seed_index.legacy_model_keys, "qwen35_4b_s4")
    # the harness's exact shape: --model-key qwen35_4b_s2 --roster-selection s2; the roster is read under the target key
    # (the suffix stripped), the pins under the seed's staged id (base64s2__<stem>)
    co = synthetic_checkout(tmp_path, objects=("none", "synth__lora_r4", "prompted__si_constitution"))
    co["roster"].write_text(yaml.safe_dump({"targets": {"qwen35_4b": {"views": {
        "s2": {"cells": {"synth__lora_r4": "base64s2__synth__lora_r4", "prompted__si_constitution": None}}}}}}))
    co["pins"].write_text(json.dumps({"qwen35_4b": {"base64s2__synth__lora_r4": {"object_sha256": STAGED_SHA}}}))
    b = JP.build_plans(config_path=str(co["cfg"]), model_key="qwen35_4b_s2", roster_selection="s2", evals=["xstest"], serving_path="lora_native",
                       campaign="sdfbase_full22", job_id="121240", job_name="f22s-qwen35_4b-s2", substrate="cluster_b", code_sha=synth.code_sha("a"))
    objs = {r.route["object"]: r.plan for r in b.runs}
    assert set(objs) == {"none", "synth__lora_r4", "prompted__si_constitution"} and not b.skipped, b.skipped
    assert objs["synth__lora_r4"]["object_id"] == STAGED_SHA and objs["synth__lora_r4"]["transfer_kind"] == "base_to_instruct"
    assert objs["none"]["transfer_kind"] == "none"
    for p in objs.values():
        assert p["model_id"] == "qwen35_4b_it" and p["extra"]["model_key"] == "qwen35_4b_s2" and p["extra"]["view_legacy"] == "s2"
        assert p["cluster_job_id"] == "121240" and p["submit_evidence_uri"] == "slurm://job/121240"
    assert "model qwen35_4b_s2 -> qwen35_4b_it (base_to_instruct)" in b.notes


def test_a_config_outside_the_checkout_needs_the_seeds_and_pins_named(tmp_path, seed_index):
    """The per-cell registration (the gsm8k rows) registers from a NARROWED copy of the overlay written beside the pending file
    — under the shared root, OUTSIDE the checkout. ``_checkout_root`` walks up from the CONFIG's path for ``configs/registry`` and falls
    back to the config's grandparent, so from that copy the seeds resolve to ``<shared>/registry/configs/registry`` and the plan refuses
    by name (both u0 waves' gsm8k rows registered nothing this way, 2026-09-30) — and had the seeds resolved, the pins off the same root
    would silently read as none. With the checkout's seeds, roster and pins NAMED the copy plans exactly what the original does."""
    co = synthetic_checkout(tmp_path)
    shared = tmp_path / "shared"
    pend = shared / "registry" / "pending"
    pend.mkdir(parents=True)
    copy = pend / "4242.config.yaml"
    copy.write_text(co["cfg"].read_text())
    kw = dict(model_key="qwen35_4b", evals=["petri_self_preservation"], serving_path="lora_native", campaign="sdfbase_full22",
              job_id="4242", job_name="g8k-qwen35_4b-none", substrate="cluster_a", code_sha=synth.code_sha("a"))
    with pytest.raises(JP.JobPlanRefused, match=r"no registry seeds at .*/shared/registry/configs/registry"):
        JP.build_plans(config_path=str(copy), **kw)
    named = JP.build_plans(config_path=str(copy), seeds_dir=str(SEEDS), roster_path=str(co["roster"]), pins_path=str(co["pins"]), **kw)
    ref = JP.build_plans(config_path=str(co["cfg"]), **kw)
    key = lambda b: sorted((r.route["eval_id"], r.route["object"], r.plan["object_id"]) for r in b.runs)   # noqa: E731
    assert len(ref.runs) == 3 and key(named) == key(ref), (key(named), key(ref))
    assert dict(named.skipped) == dict(ref.skipped)
