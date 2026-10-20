"""The multi-turn TARGET's registry row of a front-door model must be SENDABLE by the driver that asks for it -- the driver's own
predicate, run on the exact ref the plan composes, against the COMMITTED export (paper-companion-repo Stage E, 2026-09-30; ruling
`decode-multiturn-vendor-row-sends-six-keys-2026-09-30` in configs/registry/decode_policy_seed.yaml).

Every multi-turn target unit died after its ~20-minute standup. Behind the plan's unversioned ref sat a second refusal that the right
ref alone would have reached: `target_decode.registry_vendor_row` refuses a vendor row that
does not SEND all six sampling keys (`VENDOR_ROW_SENT_KEYS = decode_policy.PARAM_KEYS`; presence_penalty is not server-defaultable), and
the two pipeline-test models' `<model>|multi_turn|off|vendor_preset@v1` rows sent five (`temperature, top_p, top_k, max_tokens, seed`) --
their shipped generation_config.json names exactly temperature / top_p / top_k. The only model whose multi-turn battery had ever run the
registry path is qwen35_4b_it, whose row states the six from the model card; the paper models' filed waves passed no policy ref at all.

The rows now state and send the three keys the file does not name at the engine's own defaults (min_p 0.0, repetition_penalty 1.0,
presence_penalty 0.0 -- what a request that omits them resolves to on vLLM, and what the rows already recorded under `params_not_sent`),
so the decoded sampler is unchanged and the request, not the server, pins it. This module pins, in execution form:

  * the driver's predicate admits the row on the plan's exact ref, for both test models AND for the one existing registry-path model;
  * the sent numbers = the shipped file's three + the documented engine defaults for the rest -- no invented number;
  * the seed and the committed export agree on the rows (an edit without `registry decode-policy export` is caught);
  * the driver's real resolver (`jailbreak_run._resolve_multiturn_target_decode`) runs the registry path end to end on that ref;
  * an unversioned ref, a `@v2` ref and another model's ref are still refused BY NAME (the check that caught the wave stays);
  * KNOWN-OPEN pin: which front-door models still have no sendable multi-turn vendor row (the paper models, whose multi-turn results are
    their FILED waves) -- named, so a stage-06 plan for one of them is a known refusal, never a surprise after a standup.

Synthetic worlds only; every model name is a repo model_id."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C
from sisteer.eval.attacks import target_decode as TD
from sisteer.registry import decode_policy as DP
from sisteer.registry.adapters import dimensions as D

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
import jailbreak_run as jr  # noqa: E402

SEEDS = str(REPO / "configs" / "registry")
SHIPPED = json.load(open(REPO / "configs" / "sdfbase_shipped_generation_config.json", encoding="utf-8"))["models"]
FRONT_DOOR = yaml.safe_load(open(REPO / "reproduce" / "models.yaml", encoding="utf-8"))["models"]
RULING_MT = "decode-multiturn-vendor-row-sends-six-keys-2026-09-30"
#: the front-door models of reproduce/models.yaml whose rows come from the shipped file -> the served checkpoint's HF repo. NONE
#: since 2026-10-08: the configured-not-run gemma4_31b left the public tree, and the worked example qwen35_4b_demo is a reference
#: model with no decode-policy row by design (the registry owner's ruling) — it reads as "no row" below
TEST_MODELS = {}
#: the one model whose multi-turn battery ran the registry path before this work (its row states the six from the model card)
EXISTING = "qwen35_4b_it"
#: the keys the shipped files do not name, and the value a request that omits each resolves to on vLLM (contracts.py: vLLM's neutral
#: min_p / repetition_penalty, the OpenAI-protocol presence_penalty) -- the seed states and sends exactly these, never a new number
ENGINE_DEFAULTS = {"min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}
SIX = tuple(DP.PARAM_KEYS)


@pytest.fixture(scope="module")
def pf():
    """The COMMITTED package export -- what a driver on a cluster reads (never $SISTEER_DECODE_POLICY, which another test may point
    at a fixture)."""
    return DP.PolicyFile.load(DP.PACKAGE_EXPORT_PATH)


@pytest.fixture(scope="module")
def seed_idx():
    return D.decode_policy_payloads(D.load_seeds(SEEDS))


def _ref(pf, model_key):
    """The plan's ref: the driver's own composer over the export's model-id mapping (what stage 06 puts in F22_MT_TARGET_POLICY_REF)."""
    return TD.vendor_preset_policy_ref(pf.model_id(model_key), "multi_turn")


def _row(pf, model_key):
    return TD.registry_vendor_row(model_key, "multi_turn", path=pf.path, expect_ref=_ref(pf, model_key))


def _never_yaml(*a, **k):
    raise AssertionError("vendor_preset_for (the preset yaml) must not be consulted on the registry path")


def _ns(**over):
    base = dict(attack="crescendo", multiturn_target_decode="vendor", multiturn_target_seed="per_turn", trial=0)
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------------------------
# the driver's predicate, on the plan's exact ref
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", sorted(TEST_MODELS) + [EXISTING])
def test_the_drivers_predicate_admits_the_row_on_the_plans_exact_ref(pf, model_key):
    ref = _ref(pf, model_key)
    assert ref == f"{pf.model_id(model_key)}|multi_turn|off|vendor_preset@v{TD.VENDOR_PRESET_POLICY_VERSION}"
    row = _row(pf, model_key)                                        # refuses (VendorPolicyRefused) on any defect -- the wave's death
    assert row.ref == ref == row.policy.policy_id
    assert row.policy.mode == TD.VENDOR_PRESET_MODE and row.policy.superseded_by_policy_id is None and row.policy.is_default == 1
    # every one of the six sampling keys is SENT (the battery contract), plus the cap and the seed
    assert tuple(row.decode) == SIX and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row.decode.values())
    assert set(SIX) | {"max_tokens", "seed"} <= set(row.policy.params_sent), row.policy.params_sent
    assert row.policy.request_params() == row.decode
    # what the row leaves to the server is exactly the one key the OpenAI protocol has and vLLM's sampler does not model
    assert set(row.policy.params_not_sent) == {"frequency_penalty"}, row.policy.params_not_sent
    assert TD.cap_in_max_new_tokens_rule(row.policy.max_new_tokens_rule) == 4096
    # a phase view spelling of the served model names the same row (the sbatch passes --model-key <key>, the plan the model id)
    if model_key in TEST_MODELS:
        assert TD.registry_vendor_row(f"{model_key}_u0", "multi_turn", path=pf.path, expect_ref=ref).ref == ref


@pytest.mark.parametrize("model_key", sorted(TEST_MODELS))
def test_the_test_models_send_the_shipped_files_three_numbers_and_the_engine_defaults_for_the_rest(pf, model_key):
    """No invented number: temperature / top_p / top_k are the shipped file's (pinned with its sha in
    configs/sdfbase_shipped_generation_config.json), and the three keys the file does not name are sent at the value a request that
    omits them resolves to -- so a six-key request decodes exactly as the five-key request did."""
    row = _row(pf, model_key)
    shipped = SHIPPED[TEST_MODELS[model_key]]
    assert shipped["state"] == "present" and set(shipped["sampler_keys"]) == {"temperature", "top_p", "top_k"}   # the file names none of the three
    assert row.decode == {**shipped["sampler_keys"], **ENGINE_DEFAULTS}, row.decode
    # the defaults are the documented ones, by reference to the constants that carry them
    assert {k: C.GREEDY_REFERENCE_RESOLVED[k] for k in ENGINE_DEFAULTS} == ENGINE_DEFAULTS
    assert "presence_penalty" not in C.SERVER_DEFAULTABLE_SAMPLING_KEYS      # why it must ride the request: no server fills it
    assert {"min_p", "repetition_penalty"} <= set(C.SERVER_DEFAULTABLE_SAMPLING_KEYS)
    assert row.policy.decode_source == "generation_config" and row.policy.ruling_ref == RULING_MT


@pytest.mark.parametrize("model_key", sorted(TEST_MODELS))
def test_the_seed_and_the_committed_export_agree_on_the_row(pf, seed_idx, model_key):
    """A seed edit that is not re-exported (`registry decode-policy export`) would leave the drivers on the old row; the two must say
    the same numbers, the same sent set and the same ruling -- and the loader must accept the row with no `same_values_as` (its six
    stated numbers now differ from the static row's three)."""
    ref = _ref(pf, model_key)
    pol = pf.by_ref(ref)
    payload, extra = seed_idx.rows[pol.policy_id]
    assert seed_idx.numbers(pol.policy_id) == pol.numbers() == {**SHIPPED[TEST_MODELS[model_key]]["sampler_keys"], **ENGINE_DEFAULTS}
    assert list(extra["params_sent"]) == list(pol.params_sent)
    assert set(payload["params_not_sent_json"]) == set(pol.params_not_sent) == {"frequency_penalty"}
    assert "same_values_as" not in extra and extra.get("decode_ruling_ref") == RULING_MT and payload["is_default"] == 1
    # the sibling rows of the same model keep the file's three (their harnesses send what their rows say); only multi_turn widened
    for tail in ("static|off|vendor_preset", "over_refusal|off|vendor_preset", "bon|off|vendor_preset", "petri|off|vendor_preset", "gsm8k|on|vendor_preset"):
        sib = pf.by_ref(f"{pf.model_id(model_key)}|{tail}@v1")
        assert set(sib.request_params()) == {"temperature", "top_p", "top_k"} and sib.numbers()["presence_penalty"] is None, tail
    # both committed copies of the export carry the same row (the drivers read the package copy, the tests the repo copy)
    a = DP.PolicyFile.load(DP.repo_export_path()).by_ref(ref)
    assert a.params == pol.params and a.params_sent == pol.params_sent and a.numbers() == pol.numbers()


# --------------------------------------------------------------------------------------------
# the driver's real resolver, end to end on the registry path
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", sorted(TEST_MODELS) + [EXISTING])
def test_the_drivers_own_resolver_runs_the_registry_path_on_the_plans_ref(pf, monkeypatch, model_key):
    """`jailbreak_run._resolve_multiturn_target_decode` with exactly what target_attacks.sbatch passes on a wave: --multiturn-target-decode
    vendor, --model-key, --multiturn-target-policy-ref <the plan's ref>, --multiturn-target-seed-base <s>; the yaml preset is never read;
    the six numbers on the resolved block are the row's; the cap the row names equals the cap the replies are dispatched with."""
    monkeypatch.delenv(DP.ENV_VAR, raising=False)                  # the committed export, as on a cluster
    monkeypatch.setattr(jr, "vendor_preset_for", _never_yaml)
    ref = _ref(pf, model_key)
    hf = pf.models[pf.model_id(model_key)]["hf_repo"]
    cfg = {"model": hf, "model_revision": "unverified", "attacks": {"target_max_new_tokens": 4096}}
    ns = _ns(model_key=model_key, multiturn_target_policy_ref=ref, multiturn_target_seed_base=1)
    r = jr._resolve_multiturn_target_decode(cfg, ns, served_target=True)
    row = _row(pf, model_key)
    assert r["mode"] == "vendor" and r["vendor"]["decode"] == row.decode == r["record"]["preset"]
    assert r["vendor"]["seed_base"] == 1 and r["record"]["seed_base"] == 1 and r["record"]["multiturn_target_seed_base"] == 1
    assert r["record"]["source"] == ref == r["vendor"]["source"]
    assert r["record"]["source_kind"] == jr.SOURCE_KIND_REGISTRY_ROW and r["record"]["preset_basis"] == jr.PRESET_BASIS_REGISTRY_ROW
    pol = r["record"]["decode_policy"]
    assert pol["policy_id"] == ref and pol["policy_ref_source"] == "--multiturn-target-policy-ref" and pol["superseded_by_policy_id"] is None
    assert pol["target_cap"] == 4096 == pol["cap_named_by_rule"] and pol["policy_file"] == pf.path
    assert pol["policy_set_sha256"] == pf.header["policy_set_sha256"]


# --------------------------------------------------------------------------------------------
# the check that caught the wave stays: wrong refs are refused by name
# --------------------------------------------------------------------------------------------


def test_an_unversioned_a_bumped_and_a_foreign_ref_are_still_refused_by_name(pf, monkeypatch, tmp_path):
    monkeypatch.delenv(DP.ENV_VAR, raising=False)
    good = _ref(pf, "qwen35_35b_a3b")
    for bad in ("qwen35_35b_a3b|multi_turn|off|vendor_preset",             # what the plan typed by hand on 2026-09-30 (unversioned)
                "qwen35_35b_a3b|multi_turn|off|vendor_preset@v2",          # a version bump the composer does not know
                _ref(pf, "gemma4_31b"), _ref(pf, EXISTING),                 # another model's row
                "qwen35_35b_a3b|static|off|vendor_preset@v1"):             # another segment's row
        assert bad != good
        with pytest.raises(TD.VendorPolicyRefused, match="is not the served model's multi_turn vendor row"):
            TD.registry_vendor_row("qwen35_35b_a3b", "multi_turn", path=pf.path, expect_ref=bad)
        # and through the driver: a SystemExit naming the flag, before any byte is written
        cfg = {"model": pf.models["qwen35_35b_a3b"]["hf_repo"], "model_revision": "unverified", "attacks": {"target_max_new_tokens": 4096}}
        with pytest.raises(SystemExit, match=r"--multiturn-target-policy-ref: policy ref .* is not the served model's multi_turn vendor row"):
            jr._resolve_multiturn_target_decode(cfg, _ns(model_key="qwen35_35b_a3b", multiturn_target_policy_ref=bad, multiturn_target_seed_base=1),
                                                served_target=True)
    # a row that sends fewer than the six is refused too -- the defect this module fixes, pinned as still-wrong on a synthetic export
    data = json.loads(Path(pf.path).read_text(encoding="utf-8"))
    for p in data["policies"]:
        if p["policy_id"] == good:
            p["params_sent"] = ["temperature", "top_p", "top_k", "max_tokens", "seed"]
    data["header"]["policy_set_sha256"] = DP.policy_set_sha256(data)
    five = tmp_path / "five_key_export.json"
    five.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    with pytest.raises(TD.VendorPolicyRefused, match=r"does not SEND \['min_p', 'repetition_penalty', 'presence_penalty'\]"):
        TD.registry_vendor_row("qwen35_35b_a3b", "multi_turn", path=str(five), expect_ref=good)


# --------------------------------------------------------------------------------------------
# KNOWN-OPEN pin: the front-door models with no sendable multi-turn vendor row
# --------------------------------------------------------------------------------------------


def test_front_door_models_without_a_sendable_multiturn_row_are_exactly_the_filed_paper_models(pf):
    """reproduce/models.yaml names six models; a stage-06 multi-turn plan for one whose row the driver cannot send would die the way the
    u0 wave did -- after the standup. Pinned by name: the pipeline-test models pass; the paper models (gemma4e4b_it, llama8b_it: no
    `|multi_turn|off|vendor_preset` row at all; nemotron3nano4b: a row that sends four keys) do not, and their multi-turn results in this
    work are their FILED waves (the yaml preset path, no policy ref). Onboarding another model into the front door means making this
    predicate pass for it before its first multi-turn unit is placed (the memory rule: a new model needs a canonical, SENDABLE
    instrument ref per eval family)."""
    verdict = {}
    for key, ent in FRONT_DOOR.items():
        mid = ent["serve"]["checkpoint"]["model_id"]
        try:
            _row(pf, mid)
            verdict[mid] = "sendable"
        except TD.VendorPolicyRefused as e:
            verdict[mid] = "no row" if "has no row" in str(e) else ("does not send the six" if "does not SEND" in str(e) else str(e)[:60])
        if ent.get("run_in_this_work"):
            assert verdict[mid] == "sendable", (mid, verdict[mid])       # the model this work RUNS must pass before any unit is placed
    # the worked example passes since 2026-10-09: the registry owner's follow-up seeded its `qwen35_4b_demo|multi_turn|off|vendor_preset`
    # twin of the 4B's six-key row, and the front door's owner bound the key's three cap-4096 multi-turn instruments to it (the row a fresh
    # run sends) -- the predicate this docstring asks of a model before its first multi-turn unit is placed
    assert {m for m, v in verdict.items() if v == "sendable"} == set(TEST_MODELS) | {EXISTING, "qwen35_4b_demo"}, verdict
    assert verdict == {**{m: "sendable" for m in TEST_MODELS}, EXISTING: "sendable", "qwen35_4b_demo": "sendable",
                       "gemma4e4b_it": "no row", "llama8b_it": "no row", "nemotron3nano4b": "does not send the six"}, verdict
