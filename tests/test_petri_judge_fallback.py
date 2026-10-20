"""The JUDGE fallback chain (2026-09-19): Grok 4.6 is the fallback for BOTH roles.

Every leg still running in the paired stage-B Petri wave was stalled on the JUDGE, claude-opus-5, declining
transcripts with a safety-classifier refusal (categories reasoning_extraction / general_harms) that inspect
returns as ``stop_reason == "content_filter"`` and inspect_scout.generate_answer turns into a terminal
RefusalError -- the auditor's chain never protected the judge, and the judge had nowhere to fall back to.
inspect's own ``GenerateConfig.fallback_models`` cannot name Grok: for the Anthropic provider it is a
SERVER-side fallback (extra_body["fallbacks"], first-party Claude models only). So the judge gets the auditor's
client-side chain, parameterized by role, with ONE deliberate extension: it switches legs on a refusal
RETURNED as output, which the auditor chain (pinned here to be unchanged) never did.

These pins mirror tests/test_petri_auditor_fallback.py field for field:
* ``judge_fallback_models`` is a harness-hash input WHEN NON-EMPTY and absent from the inputs otherwise, so
  every recorded run key stands (the additive rule the auditor field established);
* OMISSION means NO chain (unlike the auditor's default-on-omission, the trap that re-instated the abliterated
  Qwen from a generator that never spelled it) -- a judge chain exists only where a manifest spells it;
* the loader refuses `false` / `null` / "" / {} / a scalar, and a repeated model;
* the eval-log metadata stamps the chain only when non-empty and resume reads absent == [];
* the driver wraps the judge in a declared chain (role "judge", its own store key) and leaves it bare otherwise.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import auditor_fallback as af
from sisteer.capeval.tiers import petri_runner as pr

from test_petri_auditor_fallback import (  # noqa: E402  (the shared world + helpers)
    PROMPT_FILES, world, _cell_under, _harness_inputs, _manifest_declaring, COMMITTED_MANIFEST,
    FakeModel,
)

GROK = ["openrouter/x-ai/grok-4.6"]
OPUS48 = "anthropic/claude-opus-4-8"
#: M11: opus-4-8 BEFORE grok, for BOTH roles -- Grok 4.6 refused some of the same content
#: Opus 5 declined (a 403 permission-denied from xAI on a live probe-verified key), so a first-party leg goes first
M11_CHAIN = [OPUS48, "openrouter/x-ai/grok-4.6"]
#: the wave's own manifests: the judge chain is spelled there (and nowhere else in the tree)
WAVE_MANIFESTS = [Path(__file__).parents[1] / "configs" / "capeval" / "manifests"
                  / f"petri_full22_qwen35_4b_s{s}.yaml" for s in (1, 2, 3)] + [
    Path(__file__).parents[1] / "configs" / "seedrep" / "arditi_allpos_qwen35_4b_ar_c0p3_seedrep_manifest.yaml"]


# --- the harness identity ------------------------------------------------------------

def test_declared_judge_chain_is_a_harness_identity_input(world):
    from sisteer.capeval.petri import defense_cells as dc
    from test_petri_store_shape import _source

    chained = replace(world.spec, judge_fallback_models=GROK)
    cells = dc.enumerate_cells(chained, _source(world.root), behavior_files=world.bf,
                               prompt_files=PROMPT_FILES)
    changed = {c.cell_id: c for c in cells.all_served()}["si_safety__psr_kl_all"]
    original = world.cells["si_safety__psr_kl_all"]
    assert changed.harness_config_hash != original.harness_config_hash
    assert changed.petri_run_key != original.petri_run_key
    inputs = _harness_inputs(chained, world, changed)
    assert inputs["role_models"]["judge_fallbacks"] == GROK
    # the auditor's own entry is untouched by the judge's chain
    assert inputs["role_models"]["auditor_fallbacks"] == list(world.spec.auditor_fallback_models)


def test_empty_and_omitted_judge_chain_key_byte_identically_to_the_pre_field_inputs(world):
    """The additive rule: an empty judge chain (the dataclass default, i.e. omission) enters neither the
    inputs nor the hash -- the inputs equal the mapping the pre-field code enumerated, so every recorded
    run key stands."""
    assert world.spec.judge_fallback_models == []          # the shared world omits the field
    cell = _cell_under(world.spec, world)
    inputs = _harness_inputs(world.spec, world, cell)
    assert "judge_fallbacks" not in inputs["role_models"]
    chained = replace(world.spec, judge_fallback_models=GROK)
    pre_field = copy.deepcopy(_harness_inputs(chained, world, cell))
    del pre_field["role_models"]["judge_fallbacks"]
    assert inputs == pre_field
    assert cell.harness_config_hash == cm.sha256_hex16(cm.canonical_json(pre_field).encode("utf-8"))


# --- the manifest LOADER ---------------------------------------------------------------

def test_loader_omission_means_no_judge_chain_and_explicit_is_loaded_as_declared(tmp_path):
    omitted = _manifest_declaring(tmp_path, "\n", "omitted.yaml")
    assert cm.load_petri_spec(omitted).judge_fallback_models == []
    assert cm.load_petri_spec(COMMITTED_MANIFEST).judge_fallback_models == []   # petri_twosi spells none
    one = _manifest_declaring(tmp_path, "\n  judge_fallback_models: [openrouter/x-ai/grok-4.6]\n", "one.yaml")
    assert cm.load_petri_spec(one).judge_fallback_models == GROK
    empty = _manifest_declaring(tmp_path, "\n  judge_fallback_models: []\n", "empty.yaml")
    assert cm.load_petri_spec(empty).judge_fallback_models == []


@pytest.mark.parametrize("declaration", [
    "judge_fallback_models: false",
    "judge_fallback_models: null",
    "judge_fallback_models:",
    'judge_fallback_models: ""',
    "judge_fallback_models: {}",
    "judge_fallback_models: openrouter/x-ai/grok-4.6",   # a scalar, never a list of its characters
    "judge_fallback_models: ['']",
])
def test_loader_refuses_a_judge_value_that_is_not_a_list_of_models(tmp_path, declaration):
    path = _manifest_declaring(tmp_path, f"\n  {declaration}\n")
    with pytest.raises(cm.ManifestError, match="judge_fallback_models must be a list of non-empty model strings"):
        cm.load_petri_spec(path)


def test_validator_refuses_a_judge_chain_that_repeats_the_judge(world):
    spec = replace(world.spec, judge_fallback_models=[world.spec.judge_model])
    with pytest.raises(cm.ManifestError, match="judge chain repeats a model"):
        cm.validate_petri_spec(spec)


def test_the_wave_manifests_spell_opus48_then_grok_for_both_roles():
    """M11: the ORDER is the pin -- opus-4-8 is leg index 0 of each chain, grok index 1, nothing else, no Qwen."""
    for path in WAVE_MANIFESTS:
        spec = cm.load_petri_spec(path)
        for chain in (spec.auditor_fallback_models, spec.judge_fallback_models):
            assert chain == M11_CHAIN, (path, chain)
            assert chain[0] == OPUS48 and chain[1] == GROK[0] and len(chain) == 2
            assert not any("Huihui" in m or "abliterated" in m for m in chain)
        # the primaries are unchanged, and no leg repeats a primary
        assert spec.auditor_model == "anthropic/claude-sonnet-5" and spec.judge_model == "anthropic/claude-opus-5"
        cm.validate_petri_spec(spec)


def test_two_leg_chains_key_differently_from_one_leg_and_from_the_reversed_order(world):
    """M11 re-keys: [grok] -> [opus-4-8, grok] changes harness_config_hash and hence petri_run_key, for each role
    on its own; and the chain is ORDERED -- the reversed chain is another instrument again."""
    cell_id = "si_safety__psr_kl_all"
    reversed_chain = list(reversed(M11_CHAIN))
    for field in ("auditor_fallback_models", "judge_fallback_models"):
        one = _cell_under(replace(world.spec, **{field: GROK}), world, cell_id)
        two = _cell_under(replace(world.spec, **{field: M11_CHAIN}), world, cell_id)
        rev = _cell_under(replace(world.spec, **{field: reversed_chain}), world, cell_id)
        hashes = {one.harness_config_hash, two.harness_config_hash, rev.harness_config_hash}
        assert len(hashes) == 3, (field, hashes)
        assert len({one.petri_run_key, two.petri_run_key, rev.petri_run_key}) == 3
        key = "auditor_fallbacks" if field.startswith("auditor") else "judge_fallbacks"
        assert _harness_inputs(replace(world.spec, **{field: M11_CHAIN}), world, two)["role_models"][key] == M11_CHAIN
    # the additive property is untouched by the chain's length: omission still enters nothing
    assert "judge_fallbacks" not in _harness_inputs(world.spec, world, _cell_under(world.spec, world))["role_models"]


# --- eval-log METADATA: stamped when non-empty, absent == [] on resume ------------------

def test_metadata_stamps_the_judge_chain_only_when_declared_and_resume_reads_absent_as_empty(world):
    cell = _cell_under(world.spec, world)
    b = "self_preservation"
    bare = pr.eval_log_metadata(world.spec, cell, b, world.bf, PROMPT_FILES)
    assert pr.JUDGE_FALLBACK_METADATA_KEY not in bare
    chained_spec = replace(world.spec, judge_fallback_models=GROK)
    chained_cell = _cell_under(chained_spec, world)
    stamped = pr.eval_log_metadata(chained_spec, chained_cell, b, world.bf, PROMPT_FILES)
    assert stamped[pr.JUDGE_FALLBACK_METADATA_KEY] == GROK

    def log_with(md, spec):
        return {"eval": {"metadata": md,
                         "model_roles": {"auditor": spec.auditor_model, "judge": spec.judge_model,
                                         "target": cell.virtual_model_name}}}
    verify = lambda log, spec, c: pr.verify_log_metadata(  # noqa: E731
        log, spec=spec, cell=c, dir_key=b, behavior_files=world.bf, prompt_files=PROMPT_FILES)
    # a pre-field log (no key) and an explicit [] both verify against a chain-less manifest
    verify(log_with(bare, world.spec), world.spec, cell)
    verify(log_with({**bare, pr.JUDGE_FALLBACK_METADATA_KEY: []}, world.spec), world.spec, cell)
    # a chained log never resumes under a chain-less manifest, nor the reverse
    with pytest.raises(pr.PetriRunnerError, match=f"{pr.JUDGE_FALLBACK_METADATA_KEY}: log="):
        verify(log_with({**bare, pr.JUDGE_FALLBACK_METADATA_KEY: GROK}, world.spec), world.spec, cell)
    with pytest.raises(pr.PetriRunnerError, match=f"{pr.JUDGE_FALLBACK_METADATA_KEY}: log="):
        verify(log_with({k: v for k, v in stamped.items() if k != pr.JUDGE_FALLBACK_METADATA_KEY},
                        chained_spec), chained_spec, chained_cell)


# --- the CONTROLLER: switch on a RETURNED refusal, judge only ---------------------------

def _refusal():
    return SimpleNamespace(stop_reason="content_filter", metadata=None)


def _scored():
    return SimpleNamespace(stop_reason="stop", metadata=None)


def test_judge_chain_switches_on_a_returned_refusal_and_records_its_own_provenance():
    opus = FakeModel("anthropic/claude-opus-5", [_refusal()])
    grok = FakeModel("openrouter/x-ai/grok-4.6", [_scored()])
    store = {}
    ctl = af.AuditorFallbackController(
        [af.AuditorLeg(opus, "anthropic/claude-opus-5", "opus-5"),
         af.AuditorLeg(grok, "openrouter/x-ai/grok-4.6", "grok-4.6")],
        state_getter=lambda: ("sample", store), role="judge",
        store_key=af.JUDGE_CHAIN_STORE_KEY, switch_on_refusal_output=True)
    out = asyncio.run(ctl.generate(["transcript"]))
    assert out.stop_reason == "stop"                        # grok's verdict is what the judge receives
    assert af.AUDITOR_CHAIN_STORE_KEY not in store          # the auditor's record is untouched
    rec = store[af.JUDGE_CHAIN_STORE_KEY]
    assert rec["display"] == "opus-5 → grok-4.6 @turn 1"
    assert [s["reason"] for s in rec["switches"]] == ["refusal_output"]
    assert rec["failures"][0]["error"] == "refusal_output: stop_reason=content_filter"
    assert out.metadata[af.JUDGE_CHAIN_STORE_KEY]["current_index"] == 1
    # the switched leg stays current on the sample's next turn
    grok.outcomes.append(_scored())
    asyncio.run(ctl.generate(["transcript"]))
    assert len(opus.calls) == 1 and len(grok.calls) == 2


def test_judge_chain_returns_the_last_legs_refusal_as_is():
    """Every leg declines: the refusal is handed back unchanged (inspect_scout files it as a terminal judge
    refusal exactly as before), with the exhaustion recorded."""
    opus = FakeModel("anthropic/claude-opus-5", [_refusal()])
    grok = FakeModel("openrouter/x-ai/grok-4.6", [_refusal()])
    store = {}
    ctl = af.AuditorFallbackController(
        [af.AuditorLeg(opus, "anthropic/claude-opus-5", "opus-5"),
         af.AuditorLeg(grok, "openrouter/x-ai/grok-4.6", "grok-4.6")],
        state_getter=lambda: ("sample", store), role="judge",
        store_key=af.JUDGE_CHAIN_STORE_KEY, switch_on_refusal_output=True)
    out = asyncio.run(ctl.generate(["transcript"]))
    assert out.stop_reason == "content_filter"
    rec = store[af.JUDGE_CHAIN_STORE_KEY]
    assert rec["final_failure"]["all_legs_exhausted"] is True
    assert len(rec["failures"]) == 2


def test_auditor_chain_still_passes_a_returned_refusal_through_unchanged():
    """The auditor's instrument is byte-identical: with the switch OFF (its default) a returned refusal is a
    success as before -- no switch, no failure record, the auditor store key as ever."""
    sonnet = FakeModel("anthropic/claude-sonnet-5", [_refusal()])
    grok = FakeModel("openrouter/x-ai/grok-4.6", [_scored()])
    store = {}
    ctl = af.AuditorFallbackController(
        [af.AuditorLeg(sonnet, "anthropic/claude-sonnet-5", "sonnet-5"),
         af.AuditorLeg(grok, "openrouter/x-ai/grok-4.6", "grok-4.6")],
        state_getter=lambda: ("sample", store))
    assert ctl.role == "auditor" and ctl.store_key == af.AUDITOR_CHAIN_STORE_KEY
    assert ctl.switch_on_refusal_output is False
    out = asyncio.run(ctl.generate(["transcript"]))
    assert out.stop_reason == "content_filter" and len(grok.calls) == 0
    rec = store[af.AUDITOR_CHAIN_STORE_KEY]
    assert rec["display"] == "sonnet-5" and "failures" not in rec and rec["switches"] == []
    assert af.JUDGE_CHAIN_STORE_KEY not in store


def test_judge_factory_returns_a_judge_role_wrapper_writing_the_judge_store_key(monkeypatch):
    import sys
    import types

    class FakeInspectModel:
        def __init__(self, api, config, model_args=None):
            self.api, self.config, self.model_args, self.role = api, config, model_args or {}, None

        def _set_role(self, role):
            self.role = role

    model_mod = types.ModuleType("inspect_ai.model"); model_mod.Model = FakeInspectModel
    monkeypatch.setitem(sys.modules, "inspect_ai", types.ModuleType("inspect_ai"))
    monkeypatch.setitem(sys.modules, "inspect_ai.model", model_mod)
    underlying = FakeModel("primary", [_scored()])
    underlying.api, underlying.config, underlying.model_args = object(), object(), {}
    store = {}
    wrapped = af.make_inspect_judge_chain(
        "anthropic/claude-opus-5", GROK, legs=[af.AuditorLeg(underlying, "anthropic/claude-opus-5", "opus-5")],
        state_getter=lambda: ("sample", store))
    assert isinstance(wrapped, FakeInspectModel) and wrapped.role == "judge"
    assert wrapped._petri_controller.switch_on_refusal_output is True
    asyncio.run(wrapped.generate(["transcript"]))
    assert store[af.JUDGE_CHAIN_STORE_KEY]["display"] == "opus-5"
    assert af.AUDITOR_CHAIN_STORE_KEY not in store


def test_judge_label_is_concise():
    assert af.auditor_label("anthropic/claude-opus-5") == "opus-5"
    assert af.auditor_label("anthropic/claude-opus-4-8") == "opus-4-8"
    assert af.auditor_label("anthropic/claude-sonnet-5") == "sonnet-5"   # opus-4-8 must not shadow the others


# --- the DRIVER wiring: the judge is wrapped only in a declared chain ---------------------

def _drive_inspect_both_roles(monkeypatch, tmp_path, spec, cell):
    """test_petri_auditor_fallback._drive_inspect, extended: also records the judge chains built."""
    import sys
    import types

    roles_seen, auditor_chains, judge_chains = [], [], []
    inspect_pkg = types.ModuleType("inspect_ai")
    inspect_pkg.eval_set = lambda group, **kwargs: (True, [])

    def task_with(task, *, model_roles, metadata):
        roles_seen.append(dict(model_roles))
        return task
    inspect_pkg.task_with = task_with
    log_mod = types.ModuleType("inspect_ai.log"); log_mod.read_eval_log = lambda location: None
    model_mod = types.ModuleType("inspect_ai.model")

    class FakeGenerateConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def merge(self, other):
            return self
    model_mod.GenerateConfig = FakeGenerateConfig
    model_mod.get_model = lambda name, **kwargs: SimpleNamespace(name=name, config=None)
    bloom = types.ModuleType("petri_bloom"); bloom.bloom_audit = lambda **kwargs: object()
    for name, mod in (("inspect_ai", inspect_pkg), ("inspect_ai.log", log_mod),
                      ("inspect_ai.model", model_mod), ("petri_bloom", bloom)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pr, "_require_inspect", lambda: inspect_pkg)
    monkeypatch.setattr(pr, "require_agent_loop_entry", lambda engine, cell: None)
    monkeypatch.setattr(pr, "eval_concurrency", lambda spec, engine: 4)
    monkeypatch.setattr(pr, "install_retry_sleep_cap", lambda: None)
    monkeypatch.setattr(pr, "eval_set_bounds", lambda: {})
    monkeypatch.setattr(af, "make_inspect_auditor_chain",
                        lambda p, f: (auditor_chains.append((p, list(f))) or ("AUDITOR-CHAIN", p, tuple(f))))
    monkeypatch.setattr(af, "make_inspect_judge_chain",
                        lambda p, f: (judge_chains.append((p, list(f))) or ("JUDGE-CHAIN", p, tuple(f))))
    engine = SimpleNamespace(petri_target=lambda cell: SimpleNamespace(config=None))
    task = pr.EvalTaskSpec(behavior_id="self_preservation", dir_key="self_preservation",
                           behavior_dir=tmp_path, log_dir=tmp_path / "logs", metadata={}, options={},
                           scenario_ids=None)
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)
    return roles_seen, auditor_chains, judge_chains


def test_inspect_driver_wraps_the_judge_only_in_a_declared_chain(world, tmp_path, monkeypatch):
    bare = replace(world.spec, auditor_fallback_models=[], judge_fallback_models=[])
    roles, ac, jc = _drive_inspect_both_roles(monkeypatch, tmp_path, bare, _cell_under(bare, world))
    assert ac == [] and jc == []
    (r,) = roles
    assert r["auditor"] == bare.auditor_model and r["judge"] == bare.judge_model

    both = replace(world.spec, auditor_fallback_models=GROK, judge_fallback_models=GROK)
    roles, ac, jc = _drive_inspect_both_roles(monkeypatch, tmp_path, both, _cell_under(both, world))
    assert ac == [(both.auditor_model, GROK)] and jc == [(both.judge_model, GROK)]
    (r,) = roles
    assert r["auditor"] == ("AUDITOR-CHAIN", both.auditor_model, tuple(GROK))
    assert r["judge"] == ("JUDGE-CHAIN", both.judge_model, tuple(GROK))
