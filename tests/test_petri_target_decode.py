"""qwen-seedrep-native-battery -- the Petri TARGET decode contract.

What is pinned here and why (every assertion is on CONTENT -- bytes, keys,
values -- and every new check is exercised against a KNOWN-BAD input too):

- ``manifest.petri_harness_config_inputs`` gains ``target_decode``
  ADDITIVE-WHEN-ABSENT: a spec with no block (every existing Petri manifest)
  and a spec declaring ``mode: harness`` produce the PRE-CHANGE input mapping
  and the PRE-CHANGE ``harness_config_hash`` / ``petri_run_key`` -- pinned as
  literals computed with the pre-change ``manifest.py`` -- so every existing
  Petri store keeps its key. A ``vendor`` block enters the hash as its
  RESOLVED identity (policy id, the six sampler keys AS SENT, the per-turn
  cap, the seed rule) and re-keys; the seed base does NOT enter (one standup
  serves every seed base).
- The registry row is resolved by EXACT ref (``PolicyFile.by_ref``) and held
  not-superseded, the target model's, eval ``petri``, thinking off, mode
  ``vendor_preset``, all six keys SENT (presence_penalty included), an
  integer per-turn cap; ``resolve(model, mode="vendor_preset")`` is pinned
  as the KNOWN-WRONG lookup (it returns the IFEval row) and is never consulted.
- The per-turn seed closure ``petri_target_seed(s, behaviour, scenario, turn)``
  and the turn index off the transcript; the EXACT chat-completions body a
  seeded vendor target sends per turn (through inspect-ai's own
  ``openai_completion_params`` over the config the provider receives): the
  six keys -- ``top_k`` / ``min_p`` / ``repetition_penalty`` in ``extra_body``
  -- ``max_tokens 1024`` and the turn's seed; the default-off headers hook.
- The runner: cell manifest / rows / eval-log metadata stamps, the
  ``target-seed-<s>`` log-dir namespace, the resume refusals (other seed base,
  other contract), the unseeded-vendor and cap-mismatch refusals; harness
  rows byte-identical to before (no new keys).
- The driver: ``--target-decode`` is an ASSERTION held to the manifest (the
  standup hashed the manifest), ``--target-seed-base`` is required under
  vendor / refused under harness and rides the spec into ``run_petri``; the
  ``cells.json`` record's additive key; standup evidence from the OTHER
  contract's enumeration refuses by name.

Synthetic world throughout (``test_petri_store_shape`` / ``test_petri_run_
serving_path`` builders); the decode registry is the SYNTHETIC fixture
``tests/fixtures/registry/petri_target_decode_policy.json`` via
``$SISTEER_DECODE_POLICY``. No vLLM, no model, no API call. CPU-only."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import petri_run as prun  # noqa: E402
import test_petri_run_serving_path as sp  # noqa: E402
import test_petri_store_shape as shape  # noqa: E402
from test_petri_store_shape import world  # noqa: E402,F401

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.serving import native_residual as _nr  # noqa: E402
from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402
from sisteer.capeval.tinycap import contracts as tc  # noqa: E402
from sisteer.registry import decode_policy as dp  # noqa: E402

FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "registry"
           / "petri_target_decode_policy.json")
#: the real active-sample accessor, captured before any test patches it
_REAL_ACTIVE_SAMPLE = pr._active_sample_identity
POLICY = "qwen35_4b_it|petri|off|vendor_preset@v1"
QWEN = "Qwen/Qwen3.5-4B"
VENDOR = cm.PetriTargetDecode(mode=cm.PETRI_TARGET_DECODE_VENDOR, policy=POLICY)
HARNESS = cm.PetriTargetDecode(mode=cm.PETRI_TARGET_DECODE_HARNESS)
#: the row's six keys AS SENT: T 0.7, top_p 0.8, top_k 20,
#: min_p 0, repetition 1.0, presence 1.5
SIX = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
       "repetition_penalty": 1.0, "presence_penalty": 1.5}
IDENTITY = {"mode": "vendor", "policy_id": POLICY, "request_params": SIX,
            "max_tokens": 1024, "seed_rule": "per_scenario_turn"}
#: the shared store-shape world's keys -- the byte-for-byte pin that an absent
#: block hashes exactly as the target_decode-less manifest.py did. The world
#: leaves BOTH chains to PetriSpec's defaults (auditor: the one grok leg;
#: judge: empty, and an EMPTY chain enters nothing), so these are the
#: GOLDEN_D111 keys, re-derived from the tree at 3931984bf (cherry-picked as
#: 004e74cb4): the auditor chain became API-only, the default is a harness-hash
#: input (role_models.auditor_fallbacks), so every key moved -- the deliberate
#: consequence the commit names: a cell re-run under it is a fresh namespace.
#: A later change (1f2fd34bb, 8681c01bd) put claude-opus-4-8 AHEAD of grok in
#: both chains of the plan's Petri MANIFESTS by generator rule and re-keyed
#: THOSE cells; it changed no default, so this default world still keys as
#: GOLDEN_D111 -- pinned as such, not moved.
GOLDEN_D111 = {"si_safety__psr_kl_all": ("dd8749adec02f591", "931868a9b8f74173"),
               "none": ("48687609a258787c", "e62125447235c550"),
               "prompted__safety": ("48687609a258787c", "479ec7d9a23657f3"),
               "none__lens0": ("dd8749adec02f591", "447820031dc6c338")}
GOLDEN = GOLDEN_D111
#: the D120_CHAIN shape -- [claude-opus-4-8, grok-4.6] in BOTH roles, the chain
#: every regenerated plan manifest carries -- and the SAME world's keys under
#: it, held below under an explicit pair of chains: a non-empty judge chain is
#: a harness-hash input too (role_models.judge_fallbacks, dbf326ead), so these
#: cells are another namespace again, distinct from GOLDEN_D111's and the
#: two-leg era's.
D120_CHAIN = ["anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6"]
GOLDEN_D120 = {"si_safety__psr_kl_all": ("eb6f3c133422bfba", "a835c5eca0c05e24"),
               "none": ("3d94bf962659c7f5", "26508e6ad378c3f3"),
               "prompted__safety": ("3d94bf962659c7f5", "812f5ee7036d8af1"),
               "none__lens0": ("eb6f3c133422bfba", "b8fed00840def36a")}
#: the SAME world's keys under the earlier two-leg chain (sonnet -> grok -> the
#: abliterated Qwen), computed with the PRE-CHANGE manifest.py (identical to
#: 0c192511). Held below under an EXPLICIT two-leg chain, so the target_decode
#: additivity is still proven byte-for-byte against the original literals, and
#: the only thing that moved the defaults is the chain.
TWO_LEG_CHAIN = ["openrouter/x-ai/grok-4.6", "openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated"]
GOLDEN_TWO_LEG = {"si_safety__psr_kl_all": ("a832e360a9c909bb", "5122d2a65d0a273b"),
                  "none": ("309858bdf57f05b9", "f0c8645a5a010913"),
                  "prompted__safety": ("309858bdf57f05b9", "307ad90dc2ad29a5"),
                  "none__lens0": ("a832e360a9c909bb", "e23f4f038eea0412")}
#: the served leg every per-cell assertion below reads
CELL = "si_safety__psr_kl_all"


@pytest.fixture(autouse=True)
def _fixture_registry(monkeypatch):
    """Every resolution in this module reads the SYNTHETIC export."""
    monkeypatch.setenv(dp.ENV_VAR, str(FIXTURE))
    # the stage-1 world the serving-path helpers' native enumeration assumes
    monkeypatch.setattr(dc, "NATIVE_STAGE2_ARMED", False)
    monkeypatch.setattr(_nr, "STAGE2_POSITION_CLASSES", False)
    yield


def _inputs(spec, w, cell):
    return cm.petri_harness_config_inputs(
        spec, behavior_files=w.bf, prompt_files=shape.PROMPT_FILES,
        serving_identity=cell.serving_identity)


def _sha(inputs) -> str:
    return cm.sha256_hex16(cm.canonical_json(inputs).encode("utf-8"))


def _qwen_world(tmp_path, **over):
    return shape._build_world(tmp_path / "qwen", target_model=QWEN, **over)


def _vendor_world(tmp_path, block=VENDOR):
    return shape._build_world(tmp_path / "v", target_model=QWEN, target_decode=block)


# ---------------------------------------------------------------------------
# 1. the hash: additive when absent (golden bytes), re-keys under vendor
# ---------------------------------------------------------------------------


def test_absent_block_keeps_every_pre_change_key_byte_for_byte(world):
    """The shared world declares no block: its inputs carry no
    ``target_decode`` key and every served leg's harness hash and run key
    equal the literals the pre-change code produced."""
    # the shape world's spec leaves both chains to PetriSpec's defaults: the one grok auditor leg and NO
    # judge chain (the later re-key moved the plan manifests, not the defaults), so GOLDEN is GOLDEN_D111
    assert world.spec.auditor_fallback_models == ["openrouter/x-ai/grok-4.6"], "D111: the one-leg default"
    assert world.spec.judge_fallback_models == [] and GOLDEN is GOLDEN_D111
    # three eras, three namespaces: every hash and every run key differs pairwise, per cell
    for cid in GOLDEN:
        eras = (GOLDEN_TWO_LEG[cid], GOLDEN_D111[cid], GOLDEN_D120[cid])
        assert len({hh for hh, _ in eras}) == 3 and len({rk for _, rk in eras}) == 3, (cid, eras)
    two_leg = replace(world.spec, auditor_fallback_models=list(TWO_LEG_CHAIN))
    d120 = replace(world.spec, auditor_fallback_models=list(D120_CHAIN), judge_fallback_models=list(D120_CHAIN))
    for cid, (hh, rk) in GOLDEN.items():
        cell = world.cells[cid]
        inputs = _inputs(world.spec, world, cell)
        assert "target_decode" not in inputs
        assert inputs["role_models"]["auditor_fallbacks"] == ["openrouter/x-ai/grok-4.6"]
        assert cell.harness_config_hash == hh == _sha(inputs)
        assert cell.petri_run_key == rk == cm.petri_run_key(
            hh, cid, cell.intervention_hash)
        # the GOLDEN_TWO_LEG literals, byte for byte, under the explicit two-leg chain: the chain is
        # the ONE input that moved (an absent target_decode block still enters nothing)
        hh2, rk2 = GOLDEN_TWO_LEG[cid]
        inputs2 = _inputs(two_leg, world, cell)
        assert "target_decode" not in inputs2 and inputs2["role_models"]["auditor_fallbacks"] == TWO_LEG_CHAIN
        assert _sha(inputs2) == hh2 and cm.petri_run_key(hh2, cid, cell.intervention_hash) == rk2
        assert {k: v for k, v in inputs2.items() if k != "role_models"} == {k: v for k, v in inputs.items() if k != "role_models"}
        # the GOLDEN_D120 literals, byte for byte, under the explicit opus-4-8 -> grok pair: both chains
        # enter role_models (a non-empty judge chain is an input, an empty one is not) and nothing else moves
        hh3, rk3 = GOLDEN_D120[cid]
        inputs3 = _inputs(d120, world, cell)
        assert "target_decode" not in inputs3
        assert (inputs3["role_models"]["auditor_fallbacks"], inputs3["role_models"]["judge_fallbacks"]) == (D120_CHAIN, D120_CHAIN)
        assert "judge_fallbacks" not in inputs["role_models"]
        assert _sha(inputs3) == hh3 and cm.petri_run_key(hh3, cid, cell.intervention_hash) == rk3
        assert {k: v for k, v in inputs3.items() if k != "role_models"} == {k: v for k, v in inputs.items() if k != "role_models"}
    assert cm.petri_target_decode_mode(world.spec) == "harness"
    assert cm.petri_target_decode_identity(world.spec) is None
    assert cm.petri_target_decode_record(world.spec) is None


def test_explicit_harness_block_is_the_absent_block(world):
    """``target_decode: {mode: harness}`` enters nothing: identical inputs,
    hash and run key to the field-less spec."""
    cell = world.cells[CELL]
    explicit = replace(world.spec, target_decode=HARNESS)
    assert _inputs(explicit, world, cell) == _inputs(world.spec, world, cell)
    assert _sha(_inputs(explicit, world, cell)) == GOLDEN[CELL][0]
    assert cm.petri_target_decode_record(explicit) is None


def test_vendor_block_enters_the_hash_as_its_resolved_identity_and_rekeys(tmp_path):
    harness = _qwen_world(tmp_path)
    vendor = _vendor_world(tmp_path)
    cid = CELL
    h_cell, v_cell = harness.cells[cid], vendor.cells[cid]
    h_in, v_in = _inputs(harness.spec, harness, h_cell), _inputs(vendor.spec, vendor, v_cell)
    assert "target_decode" not in h_in
    assert v_in["target_decode"] == IDENTITY
    without = copy.deepcopy(v_in)
    del without["target_decode"]
    assert without == h_in, "the vendor block is the ONE difference"
    assert v_cell.harness_config_hash == _sha(v_in) != h_cell.harness_config_hash
    assert v_cell.petri_run_key != h_cell.petri_run_key
    assert v_cell.intervention_hash == h_cell.intervention_hash
    assert cm.petri_target_decode_record(vendor.spec) == {"mode": "vendor", "policy": POLICY}


def test_seed_base_is_not_a_hash_input(tmp_path):
    """One standup serves every seed base: seed bases 1, 2, 3 and None hash
    identically (the base rides the manifest/rows/log dir instead)."""
    w = _vendor_world(tmp_path)
    cell = w.cells[CELL]
    base = _inputs(w.spec, w, cell)
    for s in (1, 2, 3):
        seeded = replace(w.spec, target_decode=replace(VENDOR, seed_base=s))
        assert _inputs(seeded, w, cell) == base
        assert cm.petri_target_decode_record(seeded) == {"mode": "vendor", "policy": POLICY}
        assert "seed_base" not in cm.petri_target_decode_record(seeded)


# ---------------------------------------------------------------------------
# 2. resolution: exact ref, held to the target; known-bad rows refuse
# ---------------------------------------------------------------------------


def test_identity_resolves_the_exact_ref_with_its_version(world):
    spec = replace(world.spec, target_model=QWEN, target_decode=VENDOR)
    ident = cm.petri_target_decode_identity(spec)
    assert ident == IDENTITY
    assert ident["request_params"]["presence_penalty"] == 1.5  # SENT, never a server default
    assert isinstance(ident["request_params"]["top_k"], int)
    # the SAME identity through an injected PolicyFile (the seam the tests
    # and a standup-side caller share)
    pf = dp.PolicyFile.load(str(FIXTURE))
    assert cm.petri_target_decode_identity(spec, policy_file=pf) == IDENTITY


def test_resolve_by_mode_is_the_known_wrong_lookup_and_is_never_consulted(monkeypatch):
    """``resolve(model, mode="vendor_preset")`` ignores the eval and returns
    the model-wide sampling-mode row -- the IFEval row for Qwen. Pinned so
    nobody reintroduces it: the identity never calls it."""
    pf = dp.PolicyFile.load(str(FIXTURE))
    wrong = pf.resolve(QWEN, mode="vendor_preset")
    assert wrong.policy_id == "qwen35_4b_it|ifeval|off|vendor_preset@v1"
    assert wrong.policy_id != POLICY
    calls = []
    real_by_ref = dp.PolicyFile.by_ref

    def spy_by_ref(self, ref, version=None):
        calls.append(("by_ref", ref))
        return real_by_ref(self, ref, version)

    def never_resolve(self, *a, **k):
        calls.append(("resolve", a, k))
        raise AssertionError("resolve() consulted")

    monkeypatch.setattr(dp.PolicyFile, "by_ref", spy_by_ref)
    monkeypatch.setattr(dp.PolicyFile, "resolve", never_resolve)
    monkeypatch.setattr(dp.PolicyFile, "find", never_resolve)
    spec = replace(shape._spec("ab" * 8), target_model=QWEN, target_decode=VENDOR)
    assert cm.petri_target_decode_identity(spec)["policy_id"] == POLICY
    assert calls == [("by_ref", POLICY)]


@pytest.mark.parametrize("policy, phrase", [
    ("qwen35_4b_it|petri|off|vendor_absent@v1", "names 0 rows"),          # NoPolicy: never invented
    ("qwen35_4b_it|petri|off|vendor_preset@v7", "names 0 rows"),          # wrong version
    ("qwen35_4b_it|petri|off|vendor_stale@v1", "SUPERSEDED"),
    ("gemma4e4b_it|petri|off|vendor_preset@v1", "is 'gemma4e4b_it''s row"),
    ("qwen35_4b_it|petri|off|vendor_nopresence@v1", "does not SEND ['presence_penalty']"),
    ("qwen35_4b_it|petri|off|vendor_badcap@v1", "does not start with the per-turn cap"),
    ("qwen35_4b_it|petri|on|vendor_thinking@v1", "thinking_mode 'on'"),
])
def test_known_bad_rows_refuse_by_name(world, policy, phrase):
    spec = replace(world.spec, target_model=QWEN,
                   target_decode=cm.PetriTargetDecode("vendor", policy))
    with pytest.raises((cm.ManifestError, dp.NoPolicy), match=re.escape(phrase)):
        cm.petri_target_decode_identity(spec)


def test_a_row_of_another_model_refuses_even_when_the_manifest_target_is_that_models_spelling(world):
    """The row is held to the MANIFEST's target through the spellings table:
    a Qwen row against a Gemma manifest refuses naming both."""
    spec = replace(world.spec, target_model="google/gemma-4-E4B-it", target_decode=VENDOR)
    with pytest.raises(cm.ManifestError, match="is 'qwen35_4b_it''s row but the manifest's target_model 'google/gemma-4-E4B-it' is 'gemma4e4b_it'"):
        cm.petri_target_decode_identity(spec)


def test_the_registry_itself_absent_is_a_refusal_not_a_default(world, monkeypatch, tmp_path):
    monkeypatch.setenv(dp.ENV_VAR, str(tmp_path / "nowhere.json"))
    spec = replace(world.spec, target_model=QWEN, target_decode=VENDOR)
    with pytest.raises(dp.NoPolicy, match="decode policy export not found"):
        cm.petri_target_decode_identity(spec)
    # ... and the harness spec never touches the registry at all
    assert cm.petri_target_decode_identity(world.spec) is None


# ---------------------------------------------------------------------------
# 3. the manifest block: structural validation and the YAML loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block, phrase", [
    (cm.PetriTargetDecode("greedy"), "not in \\['harness', 'vendor'\\]"),
    (cm.PetriTargetDecode("harness", policy=POLICY), "carries policy"),
    (cm.PetriTargetDecode("vendor"), "requires policy"),
    (cm.PetriTargetDecode("vendor", policy="qwen35_4b_it|petri|off|vendor_preset"), "no '@v<int>' version"),
    (cm.PetriTargetDecode("vendor", policy="qwen35_4b_it|petri|vendor_preset@v1"), "four"),
    (cm.PetriTargetDecode("vendor", policy="qwen35_4b_it|bon|off|vendor_preset@v1"), "names eval 'bon', not 'petri'"),
    (cm.PetriTargetDecode("vendor", policy=POLICY, seed_base=-1), "seed_base must be None or a non-negative int"),
    (cm.PetriTargetDecode("vendor", policy=POLICY, seed_base=True), "seed_base must be None or a non-negative int"),
])
def test_validate_refuses_malformed_blocks(world, block, phrase):
    spec = replace(world.spec, target_decode=block)
    with pytest.raises(cm.ManifestError, match=phrase):
        cm.validate_petri_spec(spec)


def _manifest_yaml(tmp_path, extra: str = "") -> Path:
    doc = f"""petri:
  audit: petri-decode-loader-test
  behaviors: [self_preservation]
  cells: [none]
  scenario_bundle_hash: {'ab' * 8}
  auditor_model: anthropic/claude-sonnet-5
  judge_model: anthropic/claude-opus-5
  scenarios_model: anthropic/claude-sonnet-5
  prompt_provenance: package-pinned@petri-bloom==0.2.6
  target_model: {QWEN}
  target_revision: {shape.REV}
  serving_backend: proxy_vllm
  serving_mode: interleaved
  serving_packages: {{vllm: 0.25.1+cu129}}
  serving_launch_flags: [--enforce-eager]
  chat_template_hash: '{'12' * 8}'
  tool_parser: qwen3_xml
  serving_impl_commit: {shape.IMPL_COMMIT}
  prompted_prefix_hashes: {{}}
  bloom_options: {{max_turns: 15, enable_prefill: false, enable_rollback: false, compaction: false, realism_filter: false, epochs: 1, seed: 0}}
{extra}"""
    path = tmp_path / "manifest.yaml"
    path.write_text(doc)
    return path


def test_loader_omitted_block_is_harness_and_a_vendor_block_loads(tmp_path):
    plain = cm.load_petri_spec(_manifest_yaml(tmp_path))
    assert plain.target_decode is None
    assert cm.petri_target_decode_mode(plain) == "harness"
    vendor = cm.load_petri_spec(_manifest_yaml(
        tmp_path, f"  target_decode:\n    mode: vendor\n    policy: '{POLICY}'\n"))
    assert vendor.target_decode == cm.PetriTargetDecode("vendor", POLICY, seed_base=None)
    assert cm.petri_target_decode_identity(vendor) == IDENTITY
    harness = cm.load_petri_spec(_manifest_yaml(tmp_path, "  target_decode:\n    mode: harness\n"))
    assert harness.target_decode == cm.PetriTargetDecode("harness")
    # the generated manifests on the tree: a NON-lane manifest loads unchanged (no block = harness); the lane's seed-view manifests
    # carry the vendor block since the generator emits it from the roster knob `views.s<s>.petri_target_decode`
    manifests = Path(__file__).resolve().parents[1] / "configs" / "capeval" / "manifests"
    real = cm.load_petri_spec(manifests / "petri_full22_qwen35_4b_it.yaml")
    assert real.target_decode is None
    lane = cm.load_petri_spec(manifests / "petri_full22_qwen35_4b_s1.yaml")
    assert lane.target_decode == cm.PetriTargetDecode("vendor", POLICY, seed_base=None)


@pytest.mark.parametrize("extra, phrase", [
    ("  target_decode: null\n", "must be a non-empty mapping"),
    ("  target_decode: vendor\n", "must be a non-empty mapping"),
    ("  target_decode: {}\n", "must be a non-empty mapping"),
    ("  target_decode: [vendor]\n", "must be a non-empty mapping"),
    (f"  target_decode:\n    mode: vendor\n    policy: '{POLICY}'\n    seed_base: 1\n", "seed_base is a RUN knob"),
    (f"  target_decode:\n    mode: vendor\n    policy: '{POLICY}'\n    seeds: 1\n", "unknown key"),
    (f"  target_decode:\n    policy: '{POLICY}'\n", "missing required key 'mode'"),
    ("  target_decode:\n    mode: vendor\n", "requires policy"),
])
def test_loader_refuses_malformed_declarations(tmp_path, extra, phrase):
    with pytest.raises(cm.ManifestError, match=phrase):
        cm.load_petri_spec(_manifest_yaml(tmp_path, extra))


# ---------------------------------------------------------------------------
# 4. the seed closure and the turn index
# ---------------------------------------------------------------------------


def _seed(s, b, sid, turn):
    key = f"{s}|petri|{b}|{sid}|{turn}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def test_petri_target_seed_is_the_static_rules_shape_over_the_petri_item():
    # the closure, spelled out: sha256(f"{s}|petri|{behaviour}|{scenario}|{turn}")[:4]
    assert pr.petri_target_seed(1, "self_preservation", "scenario_alpha", 1) == _seed(1, "self_preservation", "scenario_alpha", 1) == 433512445
    assert pr.petri_target_seed(2, "self_preservation", "scenario_alpha", 1) == 239984155
    assert pr.petri_target_seed(1, "self_preservation", "scenario_alpha", 2) == 2780829188
    assert pr.petri_target_seed(1, "delusion_sycophancy", "scenario_alpha", 1) == 3969191742
    assert pr.petri_target_seed(1, "self_preservation", "scenario_beta", 1) == 760061107
    # reproduces; every axis moves it; the recorded rule name
    assert pr.petri_target_seed(3, "b", "x", 7) == pr.petri_target_seed(3, "b", "x", 7)
    seeds = {pr.petri_target_seed(s, b, sid, t) for s in (1, 2, 3) for b in ("a", "b")
             for sid in ("x", "y") for t in (1, 2, 3)}
    assert len(seeds) == 36
    assert pr.PETRI_TARGET_SEED_RULE == cm.PETRI_TARGET_SEED_RULE == "per_scenario_turn"
    assert 0 <= pr.petri_target_seed(1, "b", "x", 1) < 2 ** 32


@pytest.mark.parametrize("args, phrase", [
    ((-1, "b", "x", 1), "seed_base must be a non-negative int"),
    ((True, "b", "x", 1), "seed_base must be a non-negative int"),
    ((1, "b", "x", 0), "turn is 1-based"),
    ((1, "", "x", 1), "behaviour must be a non-empty string"),
    ((1, "b", "x|y", 1), "scenario_id must be a non-empty string without"),
    ((1, "b", None, 1), "scenario_id must be a non-empty string"),
])
def test_petri_target_seed_refuses_malformed_inputs(args, phrase):
    with pytest.raises(pr.PetriRunnerError, match=phrase):
        pr.petri_target_seed(*args)


def test_target_turn_index_counts_the_targets_prior_replies():
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem,
                                  ChatMessageTool, ChatMessageUser)

    sysm, user = ChatMessageSystem(content="s"), ChatMessageUser(content="u")
    asst = ChatMessageAssistant(content="a")
    tool = ChatMessageTool(content="t", tool_call_id="c1")
    assert pr.target_turn_index([sysm, user]) == 1
    assert pr.target_turn_index([sysm, user, asst, user]) == 2
    assert pr.target_turn_index([sysm, user, asst, tool, user, asst, user]) == 3
    # a staged prefill is an assistant message and counts
    assert pr.target_turn_index([sysm, user, asst]) == 2
    assert pr.target_turn_index("bare prompt") == 1
    assert pr.target_turn_index([{"role": "user"}, {"role": "assistant"}, {"role": "user"}]) == 2


# ---------------------------------------------------------------------------
# 5. the request body: exact fields per turn, through inspect's own params
# ---------------------------------------------------------------------------


def test_vendor_request_fields_split_config_fields_from_extra_body():
    fields, extra = pr.vendor_target_request_fields(IDENTITY)
    assert fields == {"max_tokens": 1024, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
    assert extra == {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
    assert isinstance(extra["top_k"], int) and isinstance(extra["min_p"], float)
    broken = {**IDENTITY, "request_params": {k: v for k, v in SIX.items() if k != "presence_penalty"}}
    with pytest.raises(pr.PetriRunnerError, match="sends no \\['presence_penalty'\\]"):
        pr.vendor_target_request_fields(broken)


def test_vendor_target_config_applies_the_row_and_refuses_disagreement():
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    from inspect_ai.model import GenerateConfig
    from inspect_ai.model._openai import openai_completion_params

    cfg = pr.vendor_target_config(GenerateConfig(), IDENTITY)
    assert openai_completion_params(model="petri__v", config=cfg, tools=False) == {
        "model": "petri__v", "max_tokens": 1024, "presence_penalty": 1.5,
        "temperature": 0.7, "top_p": 0.8,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}}
    # an engine-supplied config that AGREES passes; a foreign extra_body key is kept
    agree = GenerateConfig(temperature=0.7, max_tokens=1024, extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
    merged = pr.vendor_target_config(agree, IDENTITY)
    assert merged.extra_body == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}, "min_p": 0.0, "repetition_penalty": 1.0}
    for bad, phrase in [
        (GenerateConfig(temperature=1.0), "sets temperature=1.0 against the registry row's 0.7"),
        (GenerateConfig(max_tokens=2048), "sets max_tokens=2048 against the registry row's 1024"),
        (GenerateConfig(presence_penalty=0.0), "sets presence_penalty=0.0"),
        (GenerateConfig(seed=7), "model-level seed=7"),
        (GenerateConfig(extra_body={"top_k": 50}), "extra_body carries top_k=50 against the row's 20"),
        (GenerateConfig(extra_body={"min_p": 0.1}), "extra_body carries min_p=0.1"),
        (GenerateConfig(extra_body={"seed": 3}), "extra_body carries seed=3"),
    ]:
        with pytest.raises(pr.PetriRunnerError, match=phrase):
            pr.vendor_target_config(bad, IDENTITY)


def _recording_api(vmodel: str):
    from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput

    from inspect_ai._util.registry import RegistryInfo, set_registry_info

    class RecordingAPI(ModelAPI):
        def __init__(self):
            super().__init__(model_name=vmodel, base_url="http://127.0.0.1:1/v1",
                             api_key="x", api_key_vars=[], config=GenerateConfig())
            self.calls = []
            # Model.generate names its provider through the registry (ModelName)
            set_registry_info(self, RegistryInfo(type="modelapi", name="petri-recording"))

        async def generate(self, input, tools, tool_choice, config):
            self.calls.append((list(input), config))
            return ModelOutput.from_content(self.model_name, "ok")

        def connection_key(self) -> str:
            return "recording"

    return RecordingAPI()


def _seeded(monkeypatch, *, behaviour="self_preservation", seed_base=1, scenario="scenario_alpha", epoch=1,
            vmodel="petri__none"):
    from inspect_ai.model import GenerateConfig, Model

    api = _recording_api(vmodel)
    base = Model(api, pr.vendor_target_config(GenerateConfig(), IDENTITY))
    monkeypatch.setattr(pr, "_active_sample_identity", lambda: (scenario, epoch))
    target = pr.make_seeded_target(base, behaviour=behaviour, seed_base=seed_base,
                                   virtual_model=vmodel)
    return api, target


def _body(api, i, vmodel="petri__none"):
    from inspect_ai.model._openai import openai_completion_params
    return openai_completion_params(model=vmodel, config=api.calls[i][1], tools=False)


def test_seeded_target_sends_the_exact_body_per_turn(monkeypatch):
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    """The body the openai-compatible provider builds from the config it
    receives: the six keys + max_tokens + THIS turn's seed, nothing else;
    the seed moves with the turn and reproduces for an identical transcript;
    ``get_model(role="target")`` on the registered instance is the subclass."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
    from inspect_ai.model._model import get_model, init_model_roles

    api, target = _seeded(monkeypatch)
    init_model_roles({"target": target})
    resolved = get_model(role="target")
    assert resolved is target and type(resolved).__name__ == "_SeededVendorTarget"
    t1 = [ChatMessageSystem(content="sys"), ChatMessageUser(content="hi")]
    asyncio.run(resolved.generate(input=t1, tools=[]))
    t2 = t1 + [ChatMessageAssistant(content="ok"), ChatMessageUser(content="more")]
    asyncio.run(resolved.generate(input=t2, tools=[]))
    asyncio.run(resolved.generate(input=t1, tools=[]))  # the identical request again
    s1 = pr.petri_target_seed(1, "self_preservation", "scenario_alpha", 1)
    s2 = pr.petri_target_seed(1, "self_preservation", "scenario_alpha", 2)
    expected = lambda seed: {  # noqa: E731
        "model": "petri__none", "max_tokens": 1024, "presence_penalty": 1.5, "seed": seed,
        "temperature": 0.7, "top_p": 0.8,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}}
    assert _body(api, 0) == expected(s1)
    assert _body(api, 1) == expected(s2)
    assert _body(api, 2) == expected(s1)
    assert s1 != s2
    assert all(c[1].extra_headers is None for c in api.calls), "no hook: no headers"
    assert api.calls[1][0] == t2  # the transcript reaches the provider unchanged


def test_seeded_target_moves_the_seed_with_seed_base_behaviour_and_scenario(monkeypatch):
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    from inspect_ai.model import ChatMessageUser

    bodies = {}
    for s, b, sid in [(1, "self_preservation", "scenario_alpha"), (2, "self_preservation", "scenario_alpha"),
                      (1, "delusion_sycophancy", "scenario_alpha"), (1, "self_preservation", "scenario_beta")]:
        api, target = _seeded(monkeypatch, behaviour=b, seed_base=s, scenario=sid)
        asyncio.run(target.generate(input=[ChatMessageUser(content="hi")]))
        bodies[(s, b, sid)] = _body(api, 0)
        assert bodies[(s, b, sid)]["seed"] == pr.petri_target_seed(s, b, sid, 1)
    assert len({b["seed"] for b in bodies.values()}) == 4
    # the seed is the ONLY difference between the bodies
    for body in bodies.values():
        assert {k: v for k, v in body.items() if k != "seed"} == {
            "model": "petri__none", "max_tokens": 1024, "presence_penalty": 1.5,
            "temperature": 0.7, "top_p": 0.8,
            "extra_body": {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}}


def test_seeded_target_refuses_outside_a_sample_and_keeps_a_callers_config(monkeypatch):
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    api, target = _seeded(monkeypatch)
    # a per-call config the agent passes is kept (merged under the seed)
    asyncio.run(target.generate(input=[ChatMessageUser(content="hi")], config=GenerateConfig(stop_seqs=["END"])))
    assert _body(api, 0)["stop"] == ["END"] and _body(api, 0)["seed"] == pr.petri_target_seed(1, "self_preservation", "scenario_alpha", 1)
    # outside an Inspect sample there is no scenario to seed against: refuse
    # (the REAL accessor, restored: no eval is running in this process)
    monkeypatch.setattr(pr, "_active_sample_identity", _REAL_ACTIVE_SAMPLE)
    with pytest.raises(pr.PetriRunnerError, match="outside an Inspect sample"):
        asyncio.run(target.generate(input=[ChatMessageUser(content="hi")]))
    assert len(api.calls) == 1, "nothing was sent"


def test_headers_hook_is_default_off_and_rides_extra_headers_when_installed(monkeypatch):
    pytest.importorskip("inspect_ai")   # Modal CPU image: the tinycap client venv is not installed there (main's test_petri_auditor_retry_bounds idiom)
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    assert pr.TARGET_TURN_HEADERS_HOOK is None
    seen = []

    def hook(turn: pr.TargetTurn):
        seen.append(turn)
        return {tc.HEADER_TASK: turn.task, tc.HEADER_SAMPLE: tc.sample_header_value(turn.item_id, turn.sample_idx)}

    monkeypatch.setattr(pr, "TARGET_TURN_HEADERS_HOOK", hook)
    api, target = _seeded(monkeypatch, vmodel="petri__si_safety__psr_kl_all")
    t2 = [ChatMessageUser(content="hi"), ChatMessageAssistant(content="a"), ChatMessageUser(content="b")]
    asyncio.run(target.generate(input=t2))
    [turn] = seen
    assert turn == pr.TargetTurn(task="petri/self_preservation", behaviour="self_preservation",
                                 scenario_id="scenario_alpha", epoch=1, turn=2,
                                 seed=pr.petri_target_seed(1, "self_preservation", "scenario_alpha", 2),
                                 seed_base=1, virtual_model="petri__si_safety__psr_kl_all")
    assert turn.item_id == "scenario_alpha" and turn.sample_idx == 2
    assert api.calls[0][1].extra_headers == {"X-Tinycap-Task": "petri/self_preservation",
                                             "X-Tinycap-Sample": "scenario_alpha/2"}
    # the body is unchanged by the hook
    assert _body(api, 0, vmodel="petri__si_safety__psr_kl_all")["seed"] == turn.seed
    # a hook returning None attaches nothing
    monkeypatch.setattr(pr, "TARGET_TURN_HEADERS_HOOK", lambda t: None)
    api2, target2 = _seeded(monkeypatch)
    asyncio.run(target2.generate(input=[ChatMessageUser(content="hi")]))
    assert api2.calls[0][1].extra_headers is None


# ---------------------------------------------------------------------------
# 6. the runner: stamps, namespaces, refusals; harness rows byte-identical
# ---------------------------------------------------------------------------


def _run_cell(w, cell, tmp_path, seed_base, *, spec=None, tag="run"):
    # ``spec`` as given, else the WORLD's declared block with the run's seed
    # base (never a swapped policy); the store is the run key's under sweep/
    spec = spec or replace(w.spec, target_decode=replace(w.spec.target_decode, seed_base=seed_base))
    store = cs.CellStore(tmp_path / "sweep", cell.petri_run_key)
    result = pr._run(
        object(), spec, cell, store, seed=spec.bloom_options.seed,
        driver=shape._driver(w, cell), config_root=w.config_root,
        bundle_dir=w.bundle.bundle_dir, work_root=tmp_path / f"work-{tag}",
        log_root=tmp_path / "logs", prompt_files=shape.PROMPT_FILES)
    return store, result


def test_runner_stamps_the_contract_on_manifest_rows_and_log_dir(tmp_path):
    w = _vendor_world(tmp_path)
    cell = w.cells[CELL]
    store, result = _run_cell(w, cell, tmp_path, 2)
    assert store.is_done() and result["incomplete"] is False
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest[pr.TARGET_DECODE_KEY] == IDENTITY
    assert manifest[pr.TARGET_SEED_BASE_KEY] == 2
    assert pr.TARGET_DECODE_KEY not in manifest["row_constants"]
    assert manifest["row_constants"]["seed"] == 0  # the audit-wide Inspect seed, unchanged
    rows = [json.loads(line) for line in (store.cell_dir / "rows.jsonl").read_text().splitlines()]
    assert rows
    for r in rows:
        blob = json.loads(r["extra_json"])
        assert blob["target_decode"] == IDENTITY
        assert blob["sampling"] == SIX
        assert blob["seed_rule"] == "per_scenario_turn"
        assert blob["seed_base"] == 2
        assert r["task"].startswith("petri/") and r["seed"] == 0
    # completed_keys files (petri/<behaviour>, scenario_id) -- the audit's coverage pairs
    assert {(t, i) for (t, i, _m) in store.completed_keys()} == {
        (f"petri/{b}", sid) for b in shape.BEHAVIORS for sid in shape.SIDS}
    # the Inspect log dir is the run key's, sub-namespaced by the seed base
    assert (tmp_path / "logs" / cell.petri_run_key / "target-seed-2").is_dir()
    assert not any(p.name.startswith("resume-") for p in (tmp_path / "logs" / cell.petri_run_key).iterdir())


def test_runner_refuses_another_seed_base_into_the_same_store(tmp_path):
    w = _vendor_world(tmp_path)
    cell = w.cells[CELL]
    store, _ = _run_cell(w, cell, tmp_path, 1)
    before = sp._snapshot(store.cell_dir)
    with pytest.raises(pr.PetriRunnerError, match="records target decode vendor .* seed_base=1 but this run draws under vendor .* seed_base=3 .* single-contract and single-seed-base"):
        _run_cell(w, cell, tmp_path, 3, tag="again")
    assert sp._snapshot(store.cell_dir) == before, "nothing written"
    # the same seed base resumes (already complete: no relaunch)
    store2, result2 = _run_cell(w, cell, tmp_path, 1, tag="resume")
    assert result2["rows_emitted"] == 0 and result2["incomplete"] is False


def test_runner_refuses_an_unseeded_vendor_run_before_any_write(tmp_path):
    w = _vendor_world(tmp_path)
    cell = w.cells[CELL]
    store = cs.CellStore(tmp_path / "sweep", cell.petri_run_key)
    with pytest.raises(pr.PetriRunnerError, match="mode 'vendor' .* with NO seed base .* refusing an unseeded preset draw"):
        _run_cell(w, cell, tmp_path, None, spec=w.spec)
    assert not store.cell_dir.exists() or not store.manifest_path.exists()


def test_runner_refuses_a_seed_base_under_harness(world, tmp_path):
    cell = world.cells[CELL]
    spec = replace(world.spec, target_decode=cm.PetriTargetDecode("harness", seed_base=1))
    store = cs.CellStore(tmp_path / "sweep", cell.petri_run_key)
    with pytest.raises(pr.PetriRunnerError, match="seed_base 1 under mode 'harness'"):
        _run_cell(world, cell, tmp_path, None, spec=spec)
    assert not store.manifest_path.exists()


def test_runner_refuses_a_row_cap_that_is_not_the_composers(tmp_path, monkeypatch):
    """The row says 1024 per turn and ``target_response_cap`` derives 1024 from
    the composer's window; a row capping elsewhere (the fixture's 2048 row)
    refuses -- the cap is verified, never assumed."""
    cap2048 = cm.PetriTargetDecode("vendor", "qwen35_4b_it|petri|off|vendor_cap2048@v1")
    w = _vendor_world(tmp_path, cap2048)
    cell = w.cells[CELL]
    assert cm.petri_target_decode_identity(w.spec)["max_tokens"] == 2048
    assert pr.target_response_cap() == 1024
    with pytest.raises(pr.PetriRunnerError, match="caps the target at 2048 tokens per turn but the serving composer's per-turn cap is 1024"):
        _run_cell(w, cell, tmp_path, 1)


def test_harness_runner_rows_and_manifest_carry_no_new_key(world, tmp_path):
    """Today's audit byte for byte: no ``target_decode`` / ``target_seed_base``
    on the manifest, no ``target_decode`` / ``sampling`` / ``seed_rule`` /
    ``seed_base`` in any row blob, the log dir the run key's alone."""
    cell = world.cells[CELL]
    store, _ = _run_cell(world, cell, tmp_path, None, spec=world.spec)
    manifest = json.loads(store.manifest_path.read_text())
    assert pr.TARGET_DECODE_KEY not in manifest and pr.TARGET_SEED_BASE_KEY not in manifest
    for line in (store.cell_dir / "rows.jsonl").read_text().splitlines():
        blob = json.loads(json.loads(line)["extra_json"])
        assert not {"target_decode", "sampling", "seed_rule", "seed_base"} & set(blob)
    assert sorted(p.name for p in (tmp_path / "logs" / cell.petri_run_key).iterdir()) == []
    assert pr.recorded_target_decode(manifest) == (None, None)


def test_row_target_decode_fields_and_convert_kwargs(world):
    assert pr.row_target_decode_fields(None, None) == {}
    assert pr.row_target_decode_fields(IDENTITY, 3) == {
        "target_decode": IDENTITY, "sampling": SIX, "seed_rule": "per_scenario_turn", "seed_base": 3}
    for td, sb in ((IDENTITY, None), (None, 3)):
        with pytest.raises(pr.PetriRunnerError, match="half-labelled"):
            pr.row_target_decode_fields(td, sb)
    cell = world.cells[CELL]
    dk = pr.behavior_dir_key(shape.BEHAVIORS[0], cell)
    samples = [shape.sample(sid, shape.full_dims(shape.BEHAVIORS[0])) for sid in shape.SIDS]
    rows, _ = shape.convert(world, cell, dk, samples, target_decode=IDENTITY, target_seed_base=1)
    for r in rows:
        blob = json.loads(r.extra_json)
        assert blob["seed_base"] == 1 and blob["target_decode"]["policy_id"] == POLICY
    plain, _ = shape.convert(world, cell, dk, samples)
    assert all("seed_base" not in json.loads(r.extra_json) for r in plain)


@pytest.mark.parametrize("recorded, want, phrase", [
    ({}, (IDENTITY, 1), "records target decode harness .* but this run draws under vendor"),
    ({pr.TARGET_DECODE_KEY: IDENTITY, pr.TARGET_SEED_BASE_KEY: 1}, (None, None), "records target decode vendor .* but this run draws under harness"),
    ({pr.TARGET_DECODE_KEY: IDENTITY, pr.TARGET_SEED_BASE_KEY: 1}, (IDENTITY, 2), "seed_base=1 but this run draws under vendor .* seed_base=2"),
    ({pr.TARGET_DECODE_KEY: {**IDENTITY, "policy_id": "qwen35_4b_it|petri|off|vendor_stale@v1"}, pr.TARGET_SEED_BASE_KEY: 1}, (IDENTITY, 1), "vendor_stale"),
])
def test_refuse_cross_target_decode_resume_names_the_disagreement(tmp_path, recorded, want, phrase):
    manifest = {"cell_id": "x", **recorded}
    with pytest.raises(pr.PetriRunnerError, match=phrase):
        pr.refuse_cross_target_decode_resume(manifest, *want)
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (cell_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(pr.PetriRunnerError, match=phrase):
        pr.refuse_cross_target_decode_resume(cell_dir, *want)
    # agreement, and a fresh dir, are silent
    pr.refuse_cross_target_decode_resume(manifest, *pr.recorded_target_decode(manifest))
    pr.refuse_cross_target_decode_resume(tmp_path / "fresh", *want)


def test_recorded_target_decode_refuses_a_torn_record():
    with pytest.raises(pr.PetriRunnerError, match="not a vendor contract record"):
        pr.recorded_target_decode({pr.TARGET_DECODE_KEY: IDENTITY})  # no seed base
    with pytest.raises(pr.PetriRunnerError, match="not a vendor contract record"):
        pr.recorded_target_decode({pr.TARGET_DECODE_KEY: "vendor", pr.TARGET_SEED_BASE_KEY: 1})
    with pytest.raises(pr.PetriRunnerError, match="not a vendor contract record"):
        pr.recorded_target_decode({pr.TARGET_DECODE_KEY: IDENTITY, pr.TARGET_SEED_BASE_KEY: True})


def test_eval_log_metadata_stamps_and_verify_compares_by_meaning(tmp_path):
    w = _vendor_world(tmp_path)
    cell = w.cells[CELL]
    dk = pr.behavior_dir_key(shape.BEHAVIORS[0], cell)
    s1 = replace(w.spec, target_decode=replace(VENDOR, seed_base=1))
    s2 = replace(w.spec, target_decode=replace(VENDOR, seed_base=2))
    harness_spec = replace(w.spec, target_decode=None)
    md1 = pr.eval_log_metadata(s1, cell, dk, w.bf, shape.PROMPT_FILES)
    assert md1[pr.TARGET_DECODE_KEY] == IDENTITY and md1[pr.TARGET_SEED_BASE_KEY] == 1
    mdh = pr.eval_log_metadata(harness_spec, cell, dk, w.bf, shape.PROMPT_FILES)
    assert pr.TARGET_DECODE_KEY not in mdh and pr.TARGET_SEED_BASE_KEY not in mdh
    with pytest.raises(pr.PetriRunnerError, match="mode 'vendor' .* with NO seed base"):
        pr.eval_log_metadata(w.spec, cell, dk, w.bf, shape.PROMPT_FILES)
    samples = [shape.sample(sid, shape.full_dims(shape.BEHAVIORS[0])) for sid in shape.SIDS]
    log1 = shape.make_log(w, cell, dk, samples, spec=s1)
    shape._verify(w, log1, cell, dk, spec=s1)
    # another seed base's log refuses; a chain of the same manifest at seed 2 refuses seed-1 logs
    with pytest.raises(pr.PetriRunnerError, match="target_seed_base: log=1 != manifest=2"):
        shape._verify(w, log1, cell, dk, spec=s2)
    # a vendor log never verifies against a harness manifest (extra keys are
    # otherwise allowed; this one is compared by meaning)
    with pytest.raises(pr.PetriRunnerError, match="target_decode: log=.*'vendor'.* != manifest=None"):
        shape._verify(w, log1, cell, dk, spec=harness_spec)
    # a harness (pre-field) log never verifies against a vendor manifest
    logh = shape.make_log(w, cell, dk, samples, spec=harness_spec)
    assert pr.TARGET_DECODE_KEY not in logh["eval"]["metadata"]
    with pytest.raises(pr.PetriRunnerError, match="target_decode: log=None != manifest="):
        shape._verify(w, logh, cell, dk, spec=s1)
    # ... and verifies against the harness manifest exactly as before
    shape._verify(w, logh, cell, dk, spec=harness_spec)


def test_target_decode_run_identity_tolerates_specs_that_predate_the_field():
    """A caller-built namespace with no ``target_decode`` attribute is the
    harness audit it always was (the ``serving_path_of`` convention)."""
    ns = SimpleNamespace(audit="a", target_model="m")
    assert cm.petri_target_decode_of(ns) is None
    assert cm.petri_target_decode_mode(ns) == "harness"
    assert pr.target_decode_run_identity(ns) is None


# ---------------------------------------------------------------------------
# 7. the driver: the flags, the record, the standup agreement
# ---------------------------------------------------------------------------


def test_parser_spells_the_flags_on_run_and_print_cells():
    ap = prun.build_parser()
    a = ap.parse_args(["--serving-path", "lora_native", "run", "--cell-index", "0",
                       "--target-decode", "vendor", "--target-seed-base", "2"])
    assert a.target_decode == "vendor" and a.target_seed_base == 2
    b = ap.parse_args(["print-cells", "--target-decode", "harness"])
    assert b.target_decode == "harness" and not hasattr(b, "target_seed_base")
    c = ap.parse_args(["run", "--cell-id", "none"])
    assert c.target_decode is None and c.target_seed_base is None
    with pytest.raises(SystemExit):
        ap.parse_args(["run", "--cell-index", "0", "--target-decode", "greedy"])
    with pytest.raises(SystemExit):
        ap.parse_args(["print-cells", "--target-seed-base", "1"])


def test_intent_assertion_and_seed_base_application():
    vendor = sp._spec(target_model=QWEN, target_decode=VENDOR)
    harness = sp._spec(target_model=QWEN)
    ns = lambda **kw: SimpleNamespace(**kw)  # noqa: E731
    # no assertion: the manifest rules
    assert prun.check_target_decode_intent(ns(), vendor) == "vendor"
    assert prun.check_target_decode_intent(ns(target_decode=None), harness) == "harness"
    assert prun.check_target_decode_intent(ns(target_decode="vendor"), vendor) == "vendor"
    with pytest.raises(prun.DriverError, match="--target-decode vendor but the manifest's effective target decode is 'harness'"):
        prun.check_target_decode_intent(ns(target_decode="vendor"), harness)
    with pytest.raises(prun.DriverError, match="--target-decode harness but the manifest's effective target decode is 'vendor'"):
        prun.check_target_decode_intent(ns(target_decode="harness"), vendor)
    with pytest.raises(prun.DriverError, match="not in"):
        prun.check_target_decode_intent(ns(target_decode="greedy"), vendor)
    # the seed base: required under vendor, refused under harness, rides the spec
    applied = prun.apply_target_seed_base(vendor, ns(target_seed_base=3))
    assert applied.target_decode == cm.PetriTargetDecode("vendor", POLICY, seed_base=3)
    assert applied is not vendor and replace(applied, target_decode=VENDOR) == vendor
    assert prun.apply_target_seed_base(harness, ns(target_seed_base=None)) is harness
    assert prun.apply_target_seed_base(harness, ns()) is harness
    with pytest.raises(prun.DriverError, match="requires --target-seed-base"):
        prun.apply_target_seed_base(vendor, ns(target_seed_base=None))
    with pytest.raises(prun.DriverError, match="--target-seed-base 1 under target decode 'harness'"):
        prun.apply_target_seed_base(harness, ns(target_seed_base=1))
    with pytest.raises(prun.DriverError, match="non-negative int"):
        prun.apply_target_seed_base(vendor, ns(target_seed_base=-2))


def test_seed_base_changes_no_legs_key(tmp_path):
    vendor = sp._spec(target_model=QWEN, target_decode=VENDOR)
    source = sp._source(tmp_path / "bank")
    a = prun.enumerate_petri_native(vendor, source, behavior_files=sp._bf(vendor), prompt_files=shape.PROMPT_FILES)
    seeded = prun.apply_target_seed_base(vendor, SimpleNamespace(target_seed_base=2))
    b = prun.enumerate_petri_native(seeded, source, behavior_files=sp._bf(seeded), prompt_files=shape.PROMPT_FILES)
    assert [(c.cell_id, c.harness_config_hash, c.petri_run_key) for c in a.cell_set.all_served()] == \
           [(c.cell_id, c.harness_config_hash, c.petri_run_key) for c in b.cell_set.all_served()]


def test_cells_record_carries_the_declaration_additively_and_is_held_to_the_manifest(tmp_path):
    harness = sp._spec(target_model=QWEN)
    vendor = sp._spec(target_model=QWEN, target_decode=VENDOR)
    source = sp._source(tmp_path / "bank")
    for spec in (harness, vendor):
        cell_set = sp._enum_lens(spec, source)
        payload = prun._cells_payload(cell_set, spec)
        if spec is harness:
            assert set(payload) == {"audit", "expected_cells", "canonical", "controls", "absent"}
        else:
            assert payload["target_decode"] == {"mode": "vendor", "policy": POLICY}
            assert "seed_base" not in payload["target_decode"]
    h_rec = {**prun._cells_payload(sp._enum_lens(harness, source), harness), "written_at": 1.0}
    v_rec = {**prun._cells_payload(sp._enum_lens(vendor, source), vendor), "written_at": 1.0}
    prun._check_record_matches_spec(h_rec, harness, tmp_path / "cells.json")
    prun._check_record_matches_spec(v_rec, vendor, tmp_path / "cells.json")
    with pytest.raises(prun.DriverError, match="record declares target decode None but the loaded manifest declares {'mode': 'vendor'"):
        prun._check_record_matches_spec(h_rec, vendor, tmp_path / "cells.json")
    with pytest.raises(prun.DriverError, match="record declares target decode {'mode': 'vendor'.* but the loaded manifest declares None"):
        prun._check_record_matches_spec(v_rec, harness, tmp_path / "cells.json")


def _vendor_args(out_root, *, standup, table, cell_index=None, intent="vendor", seed_base=None):
    args = sp._args(out_root, serving_path=prun.SERVING_PATH_LORA_NATIVE, standup=standup,
                    table=table, cell_index=cell_index)
    args.target_decode = intent
    args.target_seed_base = seed_base
    return args


def _stub_runtime(monkeypatch, identity):
    """The runtime seam re-stubbed to record the SPEC ``run_petri`` receives
    (``sp._wire``'s recorder keeps the cell / store / manifest, not the spec);
    returns the list the specs land in."""
    seen = []

    def run_petri(engine, spec_in, cell, store, seed):
        seen.append(spec_in)
        raise sp._Ran()

    import sisteer.capeval as _cap
    rt_stub = SimpleNamespace(make_engine=lambda _ns: SimpleNamespace(proxy_identity=identity), run_petri=run_petri)
    monkeypatch.setitem(sys.modules, "sisteer.capeval.runtime", rt_stub)
    monkeypatch.setattr(_cap, "runtime", rt_stub, raising=False)
    return seen


def test_driver_end_to_end_under_native_with_a_vendor_manifest(tmp_path, monkeypatch, capsys):
    """print-cells freezes a record carrying the declaration; run holds the
    flag, requires the seed base, hands the runner a spec carrying it and a
    cell manifest stamped with the contract; a second seed base into the same
    out root refuses; standup evidence from the HARNESS enumeration refuses
    naming the hash disagreement (the reason the manifest, not the flag, is
    the authority)."""
    vendor = sp._spec(target_model=QWEN, target_decode=VENDOR)
    harness = sp._spec(target_model=QWEN)
    source, calls = sp._source(tmp_path / "bank"), []
    native_v = prun.enumerate_petri_native(vendor, source, behavior_files=sp._bf(vendor), prompt_files=shape.PROMPT_FILES)
    native_h = prun.enumerate_petri_native(harness, source, behavior_files=sp._bf(harness), prompt_files=shape.PROMPT_FILES)
    _, rec_v, table_v = sp._standup_evidence(tmp_path / "standup_v", native_v.cell_set, source, vendor, native=True)
    _, rec_h, table_h = sp._standup_evidence(tmp_path / "standup_h", native_h.cell_set, source, harness, native=True)
    sp._wire(monkeypatch, source, identity=sp.NATIVE_IDENTITY, calls=calls)
    out_root = tmp_path / "out"

    # KNOWN-BAD: the standup hashed the HARNESS manifest -> the vendor client refuses by name
    with pytest.raises(prun.DriverError, match="harness_config_hash enumerated .* but the standup served"):
        prun.cmd_print_cells(_vendor_args(out_root, standup=rec_h, table=table_h), vendor)
    assert not (out_root / prun.CELLS_JSON).exists()
    # KNOWN-BAD: the flag disagrees with the manifest
    with pytest.raises(prun.DriverError, match="--target-decode harness but the manifest's effective target decode is 'vendor'"):
        prun.cmd_print_cells(_vendor_args(out_root, standup=rec_v, table=table_v, intent="harness"), vendor)
    with pytest.raises(prun.DriverError, match="--target-decode vendor but the manifest's effective target decode is 'harness'"):
        prun.cmd_print_cells(_vendor_args(out_root, standup=rec_h, table=table_h, intent="vendor"), harness)

    assert prun.cmd_print_cells(_vendor_args(out_root, standup=rec_v, table=table_v), vendor) == 0
    record = json.loads((out_root / prun.CELLS_JSON).read_text())
    assert record["target_decode"] == {"mode": "vendor", "policy": POLICY}
    assert record["serving_path"] == prun.SERVING_PATH_LORA_NATIVE
    # the frozen vendor record refuses the harness manifest by name
    with pytest.raises(prun.DriverError, match="record declares target decode {'mode': 'vendor'"):
        prun.cmd_run(_vendor_args(out_root, standup=rec_h, table=table_h, cell_index=0, intent=None), harness)

    # run: no seed base -> refuse before the runner; nothing written
    with pytest.raises(prun.DriverError, match="requires --target-seed-base"):
        prun.cmd_run(_vendor_args(out_root, standup=rec_v, table=table_v, cell_index=2), vendor)
    assert calls == [] and not (out_root / "cells").exists()
    # run with seed base 2: the runner gets the spec carrying it; the cell
    # manifest carries the contract and the base
    with pytest.raises(sp._Ran):
        prun.cmd_run(_vendor_args(out_root, standup=rec_v, table=table_v, cell_index=2, seed_base=2), vendor)
    [call] = calls
    assert call["cell"] == sp.LORA
    manifest = call["manifest"]
    assert manifest[pr.TARGET_DECODE_KEY] == IDENTITY and manifest[pr.TARGET_SEED_BASE_KEY] == 2
    assert manifest[tc.SERVED_ENGINE_FIELD] == sp.NATIVE
    assert pr.TARGET_DECODE_KEY not in manifest["row_constants"]
    # ... and the runner-bound spec carried the seed base (captured through
    # the store's manifest; the run key is the frozen record's)
    assert call["store"].key == native_v.cell_set.all_served()[2].petri_run_key
    # a second seed-run into the SAME out root refuses (single-seed-base store)
    calls.clear()
    # (a PetriRunnerError, as the cross-engine precedent: the runner's own
    # refusal, raised by the driver before the store is opened for writing)
    with pytest.raises(pr.PetriRunnerError, match="seed_base=2 but this run draws under vendor .* seed_base=3 .* single-contract and single-seed-base"):
        prun.cmd_run(_vendor_args(out_root, standup=rec_v, table=table_v, cell_index=2, seed_base=3), vendor)
    assert calls == []
    # the same seed base resumes into it (reaches the runner)
    with pytest.raises(sp._Ran):
        prun.cmd_run(_vendor_args(out_root, standup=rec_v, table=table_v, cell_index=2, seed_base=2), vendor)
    assert [c["cell"] for c in calls] == [sp.LORA]


def test_driver_hands_the_runner_a_spec_carrying_the_seed_base(tmp_path, monkeypatch):
    vendor = sp._spec(target_model=QWEN, target_decode=VENDOR)
    source = sp._source(tmp_path / "bank")
    native_v = prun.enumerate_petri_native(vendor, source, behavior_files=sp._bf(vendor), prompt_files=shape.PROMPT_FILES)
    _, rec_v, table_v = sp._standup_evidence(tmp_path / "standup_v", native_v.cell_set, source, vendor, native=True)
    sp._wire(monkeypatch, source, identity=sp.NATIVE_IDENTITY, calls=[])
    seen = _stub_runtime(monkeypatch, sp.NATIVE_IDENTITY)
    out_root = tmp_path / "out"
    assert prun.cmd_print_cells(_vendor_args(out_root, standup=rec_v, table=table_v), vendor) == 0
    with pytest.raises(sp._Ran):
        prun.cmd_run(_vendor_args(out_root, standup=rec_v, table=table_v, cell_index=0, seed_base=3), vendor)
    [spec_in] = seen
    assert spec_in.target_decode == cm.PetriTargetDecode("vendor", POLICY, seed_base=3)
    # the EFFECTIVE native spec otherwise (flags stripped), with the same legs
    assert spec_in.serving_launch_flags == list(native_v.effective_spec.serving_launch_flags)
    assert pr.target_decode_run_identity(spec_in) == (IDENTITY, 3)


def test_harness_driver_path_is_unchanged(tmp_path, monkeypatch):
    """The serving-path test's own scenario with no flags: the record's keys
    and the runner-bound spec are the pre-change ones (the spec object
    itself under lens)."""
    spec, source, calls = sp._spec(), sp._source(tmp_path / "bank"), []
    sp._wire(monkeypatch, source, identity=sp.LENS_IDENTITY, calls=calls)
    seen = _stub_runtime(monkeypatch, sp.LENS_IDENTITY)
    out_root = tmp_path / "out"
    args = sp._args(out_root, serving_path=prun.SERVING_PATH_LENS)
    assert prun.cmd_print_cells(args, spec) == 0
    record = json.loads((out_root / prun.CELLS_JSON).read_text())
    assert set(record) == {"audit", "expected_cells", "canonical", "controls", "absent", "written_at"}
    with pytest.raises(sp._Ran):
        prun.cmd_run(sp._args(out_root, serving_path=prun.SERVING_PATH_LENS, cell_index=0), spec)
    assert seen[0] is spec
    manifest = json.loads(next((out_root / "cells").glob("*/manifest.json")).read_text())
    assert pr.TARGET_DECODE_KEY not in manifest and pr.TARGET_SEED_BASE_KEY not in manifest
