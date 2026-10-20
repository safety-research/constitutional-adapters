"""The PER-REQUEST refusal fallback of the Petri auditor and judge.

The contract: Sonnet 5 (auditor) and Opus 5 (judge) are the defaults; when a role's model declines a request
with a safety-classifier refusal, THAT request is downgraded one step (Sonnet 4.6 / Opus 4.8) and the next
returns to the default; the serving model is recorded; a rate limit never triggers it. The mechanism is the
Anthropic API's own server-side fallback (inspect ``GenerateConfig.fallback_models``), which is exactly that
contract and records the serving model on every assistant message. Two ADDITIVE manifest fields carry it,
``auditor_refusal_fallback_models`` and ``judge_refusal_fallback_models``, with four pins:

* they are NOT harness-hash inputs: the role's default is the instrument, so a view gaining them keeps every cell's
  petri_run_key (the sticky chains, which ARE hash inputs, are untouched).
* the eval-log metadata stamps them when non-empty, and resume compares them by PROVENANCE: a completed sample
  is admitted iff every serving model of its auditor and judge calls lies in default + that role's refusal fallback,
  refused by sample and model otherwise.
* the driver wires them onto the role's DEFAULT model (the primary leg of a declared sticky chain, or the bare role) as
  ``GenerateConfig(fallback_models=[bare ids])``; an undeclared field wires nothing new.
* every score row carries the sample's per-role serving-model counts, the results record sums them per behaviour and
  per cell, the footing line names the fallback-served requests, and the registry judgment carries them with the
  judge-refused count.
"""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import auditor_fallback as af
from sisteer.capeval.tiers import petri_runner as pr
from test_petri_auditor_fallback import _cell_under, _harness_inputs  # noqa: E402
from test_petri_store_shape import PROMPT_FILES, SIDS, convert, full_dims, make_log, sample, world  # noqa: F401,E402

REPO = Path(__file__).parents[1]
SONNET46 = "anthropic/claude-sonnet-4-6"
OPUS48 = "anthropic/claude-opus-4-8"
GROK46 = "openrouter/x-ai/grok-4.6-20260810"          # the IMMUTABLE dated build is what the manifest declares (the terminal judge leg)
GROK46_ALIAS = "openrouter/x-ai/grok-4.6"             # the mutable alias OpenRouter echoes in responses (one judge family)
FRONT_DOOR = "qwen35_4b_demo"                             # the front-door model of reproduce/models.yaml (configured, not run: it commits no manifest)


def _generator():
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import make_full22_configs as g
    return g


def _front_door_baseline_roster(g) -> dict:
    """The production roster + the front-door model's battery block (the single-phase recipe's one wave, the row's declared wave) exactly as 05 --step roster prints it (reproduce/lib/models.py roster_block), with
    the two per-request refusal-fallback knobs ADDED here: the printed block declares neither and the committed roster carries no
    refusal-fallback line on this tree (pinned below), so these tests exercise the renderer's pass-through of a DECLARED chain
    (the dated Grok terminal judge leg)."""
    import yaml
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22")); sys.path.insert(0, str(REPO / "reproduce" / "lib"))
    import f22_roster as FR
    import models as MT
    roster = g.load_roster(g.ROSTER)
    blk = yaml.load("targets:\n" + MT.roster_block(FRONT_DOOR, ["battery"]), Loader=FR._StrictRosterLoader)["targets"][FRONT_DOOR]
    blk["views"]["battery"][g.PETRI_AUDITOR_REFUSAL_FALLBACK_KEY] = [SONNET46]
    blk["views"]["battery"][g.PETRI_JUDGE_REFUSAL_FALLBACK_KEY] = [OPUS48, GROK46]
    roster["targets"][FRONT_DOOR] = blk
    return roster


def _render_baseline_manifest() -> Path:
    """The front-door model's battery Petri manifest WITH the two refusal-fallback chains, rendered by the generator from the printer's roster block
    into a per-process temp dir (removed at exit): the front door commits no per-view manifest (05 --step configs renders them at run time)."""
    import atexit, shutil, tempfile
    g = _generator()
    d = Path(tempfile.mkdtemp(prefix="petri_refusal_fallback_s1_")); atexit.register(shutil.rmtree, d, True)
    p = d / f"petri_full22_{FRONT_DOOR}_battery.yaml"
    p.write_text(g.render_manifest(_front_door_baseline_roster(g), FRONT_DOOR, "battery", roster_path=g.ROSTER), encoding="utf-8")
    return p


BASELINE_MANIFEST = _render_baseline_manifest()


def _amended(spec):
    return replace(spec, auditor_refusal_fallback_models=[SONNET46], judge_refusal_fallback_models=[OPUS48])


def _event(role, requested, served):
    """One inspect model event in dict form: the requested model and the model that served (the provider writes the
    fallback handoff's target onto the assistant message and the output)."""
    return {"event": "model", "role": role, "model": requested,
            "output": {"model": served, "choices": [{"message": {"role": "assistant", "model": served}}]}}


def _served_sample(sid, *, auditor=("claude-sonnet-5",), judge=("claude-opus-5",), dims=None, behavior="self_preservation"):
    s = sample(sid, dims if dims is not None else full_dims(behavior))
    s["events"] = ([_event("auditor", "anthropic/claude-sonnet-5", m) for m in auditor]
                   + [_event("target", "openai/petri__x", "petri__x")]
                   + [_event("judge", "anthropic/claude-opus-5", m) for m in judge]
                   + [{"event": "model", "role": "judge", "model": "anthropic/claude-opus-5", "output": None}])   # pending: counts nothing
    return s


# --- the manifest fields -------------------------------------------------------------

def test_the_rendered_s1_manifest_declares_the_two_refusal_fallbacks_and_nothing_else_changed():
    spec = cm.load_petri_spec(str(BASELINE_MANIFEST))
    assert spec.auditor_model == "anthropic/claude-sonnet-5" and spec.judge_model == "anthropic/claude-opus-5"
    assert spec.auditor_fallback_models == ["openrouter/x-ai/grok-4.6"] and spec.judge_fallback_models == []   # the sticky chains, untouched
    assert spec.auditor_refusal_fallback_models == [SONNET46]
    assert spec.judge_refusal_fallback_models == [OPUS48, GROK46]      # Grok 4.6 the TERMINAL judge leg after Opus 4.8


@pytest.mark.parametrize("field, bad, phrase", [
    ("auditor_refusal_fallback_models", ["openrouter/x-ai/grok-4.6"], "first-party Claude ids only"),
    ("judge_refusal_fallback_models", ["anthropic/claude-opus-5"], "names the role's default"),
    ("judge_refusal_fallback_models", [OPUS48, OPUS48], "repeats a model"),
    ("auditor_refusal_fallback_models", "anthropic/claude-sonnet-4-6", "must be a list of non-empty model strings"),
    ("auditor_refusal_fallback_models", [None], "must be a list of non-empty model strings"),
])
def test_the_loader_refuses_a_foreign_provider_the_default_a_repeat_and_a_scalar(tmp_path, field, bad, phrase):
    import yaml
    doc = yaml.safe_load(BASELINE_MANIFEST.read_text())
    doc["petri"][field] = bad
    p = tmp_path / "m.yaml"; p.write_text(yaml.safe_dump(doc, sort_keys=False))
    with pytest.raises(cm.ManifestError, match=phrase):
        cm.load_petri_spec(str(p))


def test_omitting_or_emptying_the_fields_is_no_refusal_fallback(tmp_path):
    import yaml
    doc = yaml.safe_load(BASELINE_MANIFEST.read_text())
    del doc["petri"]["auditor_refusal_fallback_models"]; doc["petri"]["judge_refusal_fallback_models"] = []
    p = tmp_path / "m.yaml"; p.write_text(yaml.safe_dump(doc, sort_keys=False))
    spec = cm.load_petri_spec(str(p))
    assert spec.auditor_refusal_fallback_models == [] and spec.judge_refusal_fallback_models == []


def test_the_refusal_fallbacks_are_not_harness_hash_inputs_so_no_cell_re_keys(world):
    """The battery cells keep their petri_run_key under the amendment -- the harness inputs are byte-identical with and
    without the two fields, while the sticky chains (the existing fields) do re-key."""
    plain = world.spec
    amended = _amended(plain)
    cell = _cell_under(plain, world)
    assert _harness_inputs(amended, world, cell) == _harness_inputs(plain, world, cell)
    sticky = replace(plain, judge_fallback_models=[OPUS48])
    assert _harness_inputs(sticky, world, cell) != _harness_inputs(plain, world, cell)


# --- the eval-log stamp and the PROVENANCE resume rule ----------------------------------

def test_the_metadata_stamps_the_fields_only_when_non_empty(world):
    cell = _cell_under(world.spec, world)
    md = pr.eval_log_metadata(world.spec, cell, "self_preservation", world.bf, PROMPT_FILES)
    assert pr.AUDITOR_REFUSAL_FALLBACK_METADATA_KEY not in md and pr.JUDGE_REFUSAL_FALLBACK_METADATA_KEY not in md
    md2 = pr.eval_log_metadata(_amended(world.spec), cell, "self_preservation", world.bf, PROMPT_FILES)
    assert md2[pr.AUDITOR_REFUSAL_FALLBACK_METADATA_KEY] == [SONNET46] and md2[pr.JUDGE_REFUSAL_FALLBACK_METADATA_KEY] == [OPUS48]
    assert {k: v for k, v in md2.items() if k not in (pr.AUDITOR_REFUSAL_FALLBACK_METADATA_KEY, pr.JUDGE_REFUSAL_FALLBACK_METADATA_KEY)} == md


def test_resume_admits_a_completed_sample_by_provenance_and_refuses_one_served_outside_the_contract(world, capsys):
    """Both cases: a log from BEFORE the amendment (no stamps) resumes under the amended manifest when every
    auditor/judge call of its samples was served by the defaults (attempt 7's samples); a sample served by the fallback
    itself is admitted too; a sample whose auditor was served by grok is refused by sample and model; and a log stamped
    with the amendment resumes under the OLD manifest only while no sample was fallback-served."""
    plain = world.spec
    amended = _amended(plain)
    cell = _cell_under(plain, world)
    b = "self_preservation"

    def verify(log, spec):
        pr.verify_log_metadata(log, spec=spec, cell=cell, dir_key=b, behavior_files=world.bf, prompt_files=PROMPT_FILES)

    defaults_only = make_log(world, cell, b, [_served_sample("s1"), _served_sample("s2", judge=("anthropic/claude-opus-5",))], spec=plain)
    assert pr.AUDITOR_REFUSAL_FALLBACK_METADATA_KEY not in defaults_only["eval"]["metadata"]
    verify(defaults_only, amended)                                               # accepted: every call inside default + fallback
    out = capsys.readouterr().out
    assert "2 completed sample(s) admitted under the amended refusal-fallback contract by provenance" in out
    served_by_fallback = make_log(world, cell, b, [_served_sample("s1", auditor=("claude-sonnet-5", "claude-sonnet-4-6"), judge=("claude-opus-4-8",))], spec=plain)
    verify(served_by_fallback, amended)                                          # the fallback IS in the amended contract
    grok = make_log(world, cell, b, [_served_sample("s1"), _served_sample("s7", epoch_models := ("claude-sonnet-5", "grok-4.6"))], spec=plain) \
        if False else make_log(world, cell, b, [_served_sample("s1"), _served_sample("s7", auditor=("claude-sonnet-5", "grok-4.6"))], spec=plain)
    with pytest.raises(pr.PetriRunnerError, match=r"resume provenance mismatch .* sample 's7' epoch 1 had its auditor role served by \['grok-4.6'\]"):
        verify(grok, amended)
    stamped = make_log(world, cell, b, [_served_sample("s1")], spec=amended)     # an amended log under the OLD manifest
    assert stamped["eval"]["metadata"][pr.JUDGE_REFUSAL_FALLBACK_METADATA_KEY] == [OPUS48]
    verify(stamped, plain)                                                       # nothing was fallback-served: admissible
    stamped_served = make_log(world, cell, b, [_served_sample("s1", judge=("claude-opus-4-8",))], spec=amended)
    with pytest.raises(pr.PetriRunnerError, match=r"judge role served by \['claude-opus-4-8'\], outside the manifest's judge contract"):
        verify(stamped_served, plain)
    # equal stamps need no walk (the equality rule stands for an unchanged contract)
    verify(make_log(world, cell, b, [_served_sample("s1", auditor=("grok-4.6",))], spec=amended), amended)


def test_sample_role_serving_models_counts_auditor_and_judge_calls_by_serving_model():
    s = _served_sample("s1", auditor=("claude-sonnet-5", "claude-sonnet-5", "claude-sonnet-4-6"), judge=("anthropic/claude-opus-4-8",))
    assert pr.sample_role_serving_models(s) == {"auditor": {"claude-sonnet-5": 2, "claude-sonnet-4-6": 1}, "judge": {"claude-opus-4-8": 1}}
    assert pr.sample_role_serving_models({"events": []}) == {"auditor": {}, "judge": {}}
    assert pr.refusal_fallback_allowed(SimpleNamespace(auditor_model="anthropic/claude-sonnet-5", judge_model="anthropic/claude-opus-5",
                                                       auditor_refusal_fallback_models=[SONNET46], judge_refusal_fallback_models=[OPUS48])) \
        == {"auditor": {"claude-sonnet-5", "claude-sonnet-4-6"}, "judge": {"claude-opus-5", "claude-opus-4-8"}}


# --- the driver wiring -----------------------------------------------------------------

def _drive(monkeypatch, tmp_path, spec, cell):
    """``_inspect_driver`` with the inspect / petri_bloom seams stubbed: returns (the model_roles each task got, the
    get_model calls, the chain-maker calls with their kwargs)."""
    roles_seen, gets, a_chains, j_chains = [], [], [], []
    inspect_pkg = types.ModuleType("inspect_ai"); inspect_pkg.eval_set = lambda group, **kwargs: (True, [])

    def task_with(task, *, model_roles, metadata):
        roles_seen.append(dict(model_roles)); return task
    inspect_pkg.task_with = task_with
    log_mod = types.ModuleType("inspect_ai.log"); log_mod.read_eval_log = lambda location: None
    model_mod = types.ModuleType("inspect_ai.model")

    class FakeGenerateConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def merge(self, other):
            return self
    model_mod.GenerateConfig = FakeGenerateConfig

    class FakeModel(SimpleNamespace):
        def _set_role(self, role):
            self.role = role

    def get_model(name, **kwargs):
        gets.append((name, kwargs)); return FakeModel(name=name, config=kwargs.get("config"))
    model_mod.get_model = get_model
    bloom = types.ModuleType("petri_bloom"); bloom.bloom_audit = lambda **kwargs: object()
    for name, mod in (("inspect_ai", inspect_pkg), ("inspect_ai.log", log_mod), ("inspect_ai.model", model_mod), ("petri_bloom", bloom)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pr, "_require_inspect", lambda: inspect_pkg)
    monkeypatch.setattr(pr, "require_agent_loop_entry", lambda engine, cell: None)
    monkeypatch.setattr(pr, "eval_concurrency", lambda spec, engine: 4)
    monkeypatch.setattr(pr, "install_retry_sleep_cap", lambda: None)
    monkeypatch.setattr(pr, "eval_set_bounds", lambda: {})
    monkeypatch.setattr(af, "make_inspect_auditor_chain", lambda p, f, **kw: (a_chains.append((p, list(f), kw)) or ("AUDITOR-CHAIN", p, tuple(f))))
    monkeypatch.setattr(af, "make_inspect_judge_chain", lambda p, f, **kw: (j_chains.append((p, list(f), kw)) or ("JUDGE-CHAIN", p, tuple(f))))
    # the CLIENT-side per-request refusal fallback wrapper: recorded, never a real inspect Model here
    monkeypatch.setattr(af, "make_refusal_fallback_model",
                        lambda primary, name, fallbacks, *, role: FakeModel(kind="REFUSAL-FALLBACK", primary=primary, name=name,
                                                                            fallbacks=list(fallbacks), role=role))
    engine = SimpleNamespace(petri_target=lambda cell: SimpleNamespace(config=None))
    task = pr.EvalTaskSpec(behavior_id="self_preservation", dir_key="self_preservation", behavior_dir=tmp_path,
                           log_dir=tmp_path / "logs", metadata={}, options={}, scenario_ids=None)
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)
    return roles_seen, gets, a_chains, j_chains


def test_the_driver_puts_the_refusal_fallback_on_the_default_model_of_each_role(world, tmp_path, monkeypatch):
    """The baseline (battery) shape: a sticky auditor chain [grok] (its primary leg gets the client-side refusal fallback) and no judge
    chain (the bare judge becomes the client-side wrapper around the bare model, stamped with its role). No model is
    resolved with a config: the server-side `fallbacks` parameter is never sent again."""
    spec = _amended(replace(world.spec, auditor_fallback_models=["openrouter/x-ai/grok-4.6"], judge_fallback_models=[]))
    roles, gets, a_chains, j_chains = _drive(monkeypatch, tmp_path, spec, _cell_under(spec, world))
    (p, f, kw), = a_chains
    assert (p, f) == (spec.auditor_model, ["openrouter/x-ai/grok-4.6"]) and kw == {"primary_refusal_fallback": [SONNET46]}
    assert j_chains == []
    (r,) = roles
    assert r["auditor"] == ("AUDITOR-CHAIN", spec.auditor_model, ("openrouter/x-ai/grok-4.6",))
    assert (r["judge"].kind, r["judge"].name, r["judge"].fallbacks, r["judge"].role) == ("REFUSAL-FALLBACK", spec.judge_model, [OPUS48], "judge")
    assert r["judge"].primary.name == spec.judge_model and r["judge"].primary.role == "judge"
    assert [(n, kw) for n, kw in gets if n == spec.judge_model] == [(spec.judge_model, {"memoize": False})]
    assert not any("config" in kw for _n, kw in gets), gets       # no GenerateConfig(fallback_models=...) anywhere


def test_without_the_fields_the_driver_wires_exactly_what_it_always_did(world, tmp_path, monkeypatch):
    spec = replace(world.spec, auditor_fallback_models=["openrouter/x-ai/grok-4.6"], judge_fallback_models=[])
    roles, gets, a_chains, j_chains = _drive(monkeypatch, tmp_path, spec, _cell_under(spec, world))
    assert a_chains == [(spec.auditor_model, ["openrouter/x-ai/grok-4.6"], {})] and j_chains == []
    (r,) = roles
    assert r["judge"] == spec.judge_model and not [n for n, _ in gets if n == spec.judge_model]
    both = _amended(replace(world.spec, auditor_fallback_models=[], judge_fallback_models=[OPUS48]))
    roles, gets, a_chains, j_chains = _drive(monkeypatch, tmp_path, both, _cell_under(both, world))
    assert a_chains == [] and [(p, f, kw) for p, f, kw in j_chains] == [(both.judge_model, [OPUS48], {"primary_refusal_fallback": [OPUS48]})]
    (r,) = roles
    assert (r["auditor"].kind, r["auditor"].name, r["auditor"].fallbacks, r["auditor"].role) == ("REFUSAL-FALLBACK", both.auditor_model, [SONNET46], "auditor")
    assert not any("config" in kw for _n, kw in gets), gets
    assert not hasattr(pr, "_refusal_fallback_config") and not hasattr(af, "with_primary_config")   # the 400 path is gone


def _fake_inspect_model_module(monkeypatch, calls):
    """A stand-in ``inspect_ai.model`` with the two names the wrapper imports: a ``Model`` base (api, config, model_args)
    and ``get_model`` recording its calls."""
    model_mod = types.ModuleType("inspect_ai.model")

    class Model:
        def __init__(self, api, config, model_args=None):
            self.api, self.config, self.model_args = api, config, model_args or {}

        def _set_role(self, role):
            self.role = role

    class FakeModel(Model):
        def __init__(self, name, answers=()):
            super().__init__(api=f"api:{name}", config=f"cfg:{name}")
            self.name, self.answers, self.seen = name, list(answers), []

        async def generate(self, input, tools=(), tool_choice=None, config=None, cache=None):
            self.seen.append((input, tools, tool_choice, config, cache))
            return self.answers.pop(0)
    model_mod.Model = Model
    model_mod.FakeModel = FakeModel
    model_mod.get_model = lambda name, **kwargs: (calls.append((name, kwargs)) or FakeModel(name))
    monkeypatch.setitem(sys.modules, "inspect_ai.model", model_mod)
    return model_mod


def _output(model, stop_reason, *, details=None, metadata=None):
    choice = SimpleNamespace(stop_reason=stop_reason, stop_details=details, message=SimpleNamespace(role="assistant", model=model))
    return SimpleNamespace(model=model, stop_reason=stop_reason, choices=[choice], metadata=metadata)


def test_the_primary_leg_alone_carries_the_client_side_refusal_fallback_through_the_sticky_chain(monkeypatch):
    """`with_primary_refusal_fallback` wraps ONLY the primary leg (the pinned resolver signature is untouched); a
    non-first-party primary, or no declared fallback, leaves the legs as resolved."""
    calls = []
    mm = _fake_inspect_model_module(monkeypatch, calls)
    legs = (af.AuditorLeg(mm.FakeModel("p"), "anthropic/claude-sonnet-5", "sonnet-5"), af.AuditorLeg(mm.FakeModel("g"), "openrouter/x-ai/grok-4.6", "grok-4.6"))
    out = af.with_primary_refusal_fallback(legs, [SONNET46], role="auditor")
    assert [leg.model_name for leg in out] == ["anthropic/claude-sonnet-5", "openrouter/x-ai/grok-4.6"] and out[1] is legs[1]
    assert calls == [(SONNET46, {"memoize": False})]                                   # the fallback resolved, no config
    w = out[0].model
    assert type(w).__name__ == "RefusalFallbackModel" and w._petri_primary is legs[0].model and w._petri_primary_name == "anthropic/claude-sonnet-5"
    assert [n for n, _m in w._petri_fallbacks] == [SONNET46] and w.role == "auditor" and w._petri_fallbacks[0][1].role == "auditor"
    assert (w.api, w.config) == ("api:p", "cfg:p") and out[0].label == "sonnet-5"
    assert af.with_primary_refusal_fallback(legs[1:], [SONNET46]) == legs[1:] and af.with_primary_refusal_fallback(legs, []) == legs and len(calls) == 1
    import inspect
    assert list(inspect.signature(af.resolve_inspect_auditor_models).parameters) == ["primary_model", "fallback_models", "role"]


def test_the_client_side_refusal_fallback_reissues_once_per_fallback_on_a_returned_refusal_and_stamps_the_hand_off(monkeypatch):
    """The wrapper's contract: a non-refusal is returned as-is and no fallback is called; a RETURNED refusal
    (stop_reason content_filter) re-issues the IDENTICAL request to the fallback and returns its output with the hand-off
    stamped (requested, served, the refused attempt's stop evidence, client_side); every fallback refusing returns the
    LAST refusal, stamped all_refused; the next request is at the default again."""
    import asyncio
    calls = []
    mm = _fake_inspect_model_module(monkeypatch, calls)
    refusal = _output("claude-sonnet-5", "content_filter", details=SimpleNamespace(type="refusal", category="cyber", model_dump=lambda mode: {"type": "refusal", "category": "cyber"}))
    ok1, ok2, ok3 = _output("claude-sonnet-5", "stop"), _output("claude-sonnet-4-6", "stop", metadata={"kept": 1}), _output("claude-sonnet-5", "tool_calls")
    primary = mm.FakeModel("anthropic/claude-sonnet-5", answers=[ok1, refusal, ok3, refusal])
    w = af.make_refusal_fallback_model(primary, "anthropic/claude-sonnet-5", [SONNET46, "anthropic/claude-sonnet-4-5"], role="auditor")
    fb1, fb2 = (m for _n, m in w._petri_fallbacks)
    fb1.answers = [ok2, _output("claude-sonnet-4-6", "content_filter")]; fb2.answers = [_output("claude-sonnet-4-5", "content_filter", details=None)]
    assert [n for n, _k in calls] == [SONNET46, "anthropic/claude-sonnet-4-5"] and all(k == {"memoize": False} for _n, k in calls)
    args = ("INPUT", ("tool",), "auto", "CFG", True)
    run = lambda: asyncio.run(w.generate(*args))
    assert run() is ok1 and fb1.seen == [] and ok1.metadata is None                         # (1) no refusal: as-is, no fallback call
    out = run()                                                                              # (2) refused -> the first fallback serves
    assert out is ok2 and fb1.seen == [args] and primary.seen == [args, args]
    assert out.metadata == {"kept": 1, af.REFUSAL_FALLBACK_SERVED_KEY: {
        "requested": "anthropic/claude-sonnet-5", "served": SONNET46, "outcome": "served", "client_side": True,
        "attempts": [{"model": "anthropic/claude-sonnet-5", "stop_reason": "content_filter", "stop_details": {"type": "refusal", "category": "cyber"}}]}}
    assert run() is ok3 and len(fb1.seen) == 1                                               # (3) the next request is at the default again
    out = run()                                                                              # (4) every fallback refuses -> the LAST refusal, stamped
    assert out.model == "claude-sonnet-4-5" and out.stop_reason == "content_filter" and fb2.seen == [args]
    stamp = out.metadata[af.REFUSAL_FALLBACK_SERVED_KEY]
    assert (stamp["outcome"], stamp["served"], [a["model"] for a in stamp["attempts"]]) == ("all_refused", None, ["anthropic/claude-sonnet-5", SONNET46, "anthropic/claude-sonnet-4-5"])
    assert stamp["attempts"][2]["stop_details"] is None
    # the predicate: the two refusal shapes, and nothing else
    assert af.is_refusal_output(_output("m", "content_filter")) and af.is_refusal_output(_output("m", "content_filter", details=SimpleNamespace(type="refusal")))
    assert not af.is_refusal_output(_output("m", "stop")) and not af.is_refusal_output(_output("m", "max_tokens")) and not af.is_refusal_output(None)
    # no declared fallback: the wrapper is a pass-through even on a refusal
    bare = af.make_refusal_fallback_model(mm.FakeModel("x", answers=[refusal]), "anthropic/x", [], role="judge")
    assert asyncio.run(bare.generate("i")) is refusal and refusal.metadata is None


class _ContentCorrelatedDiscriminator:
    """Every failure classifies content-correlated with one healthy sibling: the condition under which the sticky chain
    would switch to Grok."""
    def classify_failure(self, model, sample_key, started, completed, **kwargs):
        return af.LiveDiscriminatorEvidence(kind="content_correlated", model=model, window_start=started, window_end=completed,
                                            monotonic_start=started, monotonic_end=completed, siblings_succeeded=1, siblings_failed=0,
                                            sibling_success_timestamps=((started + completed) / 2,), sibling_failure_timestamps=(),
                                            basis="synthetic: a primary sibling completed inside the window")

    def record_success(self, *a, **k):
        pass


@pytest.mark.parametrize("cause, status, etype", [
    (RuntimeError("Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'This request would exceed the rate limit'}}"), 429, "rate_limit_error"),
    (RuntimeError("Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'unsupported'}}"), 400, "invalid_request_error"),
    (ConnectionError("Connection error."), None, None),
])
def test_a_fallback_legs_own_error_is_named_and_never_rescued_by_the_sticky_chain(monkeypatch, cause, status, etype):
    """This sample's primary RETURNS a refusal, the declared fallback leg RAISES (429 at the run's
    concurrency; 400; a transport error) while a primary sibling succeeded. The wrapper raises RefusalFallbackLegError naming
    the leg and its HTTP class (marker first); the sticky chain refuses the switch — Grok is never called, the record's
    final_failure is kind hard_error with the leg and status — and the error propagates (the sample errors: the runner's
    hard-error ledger, never a scored result)."""
    import asyncio
    calls = []
    mm = _fake_inspect_model_module(monkeypatch, calls)
    refusal = _output("claude-sonnet-5", "content_filter", details=SimpleNamespace(type="refusal", category="cyber", model_dump=lambda mode: {"type": "refusal", "category": "cyber"}))
    primary = mm.FakeModel("anthropic/claude-sonnet-5", answers=[refusal])
    wrapped = af.make_refusal_fallback_model(primary, "anthropic/claude-sonnet-5", [SONNET46], role="auditor")
    (fb_name, fb), = wrapped._petri_fallbacks
    fb.answers = [cause]

    class Raising(type(fb)):
        pass
    async def raising_generate(input, tools=(), tool_choice=None, config=None, cache=None):
        fb.seen.append((input, tools, tool_choice, config, cache)); raise fb.answers.pop(0)
    fb.generate = raising_generate
    grok = mm.FakeModel("openrouter/x-ai/grok-4.6", answers=[_output("grok-4.6", "stop")])
    legs = (af.AuditorLeg(wrapped, "anthropic/claude-sonnet-5", "sonnet-5"), af.AuditorLeg(grok, "openrouter/x-ai/grok-4.6", "grok-4.6"))
    store = {}
    ticks = iter(float(x) for x in range(1, 100))
    controller = af.AuditorFallbackController(legs, discriminator=_ContentCorrelatedDiscriminator(), clock=lambda: next(ticks),
                                              state_getter=lambda: ("sample-1", store))
    args = ("INPUT", ("tool",), "auto", "CFG")
    with pytest.raises(af.RefusalFallbackLegError) as ei:
        asyncio.run(controller.generate(*args))
    err = ei.value
    assert (err.leg_model, err.primary_model, err.status, err.error_type) == (SONNET46, "anthropic/claude-sonnet-5", status, etype)
    assert str(err).startswith(af.FALLBACK_LEG_ERROR_MARKER + ": the declared per-request fallback " + SONNET46) and "never a sticky switch" in str(err)
    assert ("HTTP %d" % status if status is not None else "a transport error") in str(err) and err.__cause__ is cause
    assert [a["model"] for a in err.attempts] == ["anthropic/claude-sonnet-5"]
    assert fb.seen == [("INPUT", ("tool",), "auto", "CFG", None)] and grok.seen == []                 # identical re-issue; Grok never called
    chain = store[af.AUDITOR_CHAIN_STORE_KEY]
    assert chain["switches"] == [] and chain["current_index"] == 0 and chain["display"] == "sonnet-5"
    (failure,) = chain["failures"]
    assert (failure["fallback_leg"], failure["hard_error"], failure["status"], failure["model"]) == (SONNET46, True, status, "anthropic/claude-sonnet-5")
    assert failure["error"].startswith("RefusalFallbackLegError: " + af.FALLBACK_LEG_ERROR_MARKER)      # the marker survives the 400-char clip
    assert chain["final_failure"]["kind"] == af.FALLBACK_LEG_HARD_ERROR_KIND and chain["final_failure"]["all_legs_exhausted"] is False
    assert af.chain_kind(chain, refused=True) == af.CHAIN_UNAVAILABLE                                 # never "all legs refused"
    assert pr.FALLBACK_LEG_ERROR_MARKER == af.FALLBACK_LEG_ERROR_MARKER


def test_the_converter_files_a_fallback_leg_error_as_a_hard_error_whatever_its_status(world):
    """The sample the controller's refusal leaves behind: the primary's refusal event, the fallback's errored event (a 429 —
    a retryable status that would otherwise read as an exhausted ladder), the sample error a ModelGenerateError wrapping the
    RefusalFallbackLegError. The marker rule comes first: a HARD error (status 429 recorded), entries without a shed, no row,
    nothing terminal; the error tail starts at the leg error so the leg and its status lead."""
    from test_petri_store_shape import BEHAVIORS, convert
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    leg_text = (af.FALLBACK_LEG_ERROR_MARKER + ": the declared per-request fallback anthropic/claude-sonnet-4-6 (for anthropic/claude-sonnet-5) "
                "failed with HTTP 429 rate_limit_error — transport/HTTP, not content; one re-issue, no ladder, never a sticky switch: "
                "RuntimeError: Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error'}}")
    smp = sample(SIDS[0], None, error={"message": "ModelGenerateError('\nRequest:\n{\"messages\": [{\"role\": \"user\", \"content\": \"timed out overloaded\"}]}\n\nRefusalFallbackLegError(" + repr(leg_text) + ")')"})
    smp["uuid"] = f"synthetic-{SIDS[0]}"
    smp["events"] = [_event("auditor", "anthropic/claude-sonnet-5", "claude-sonnet-5"),
                     {"event": "model", "role": "auditor", "model": SONNET46, "timestamp": "2026-10-01T08:31:28+00:00",
                      "error": {"message": "Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error'}}"}}]
    smp["completed_at"] = "2026-10-01T08:31:31+00:00"
    exhausted, status, etype, why = pr.ladder_exhaustion(smp)
    assert (exhausted, status, etype) == (False, 429, "rate_limit_error") and "fallback leg itself failed" in why
    assert pr._error_tail(pr._sample_error_text(smp)).startswith("RefusalFallbackLegError(")
    rows, entries = convert(world, cell, b, [smp, sample(SIDS[1], full_dims(b))])
    assert len(rows) == 3 and all(e.hard_error is not None and e.shed is None for e in entries) and len(entries) == 3
    assert entries[0].hard_error.status == 429 and entries[0].hard_error.error.startswith("RefusalFallbackLegError(")
    assert entries[0].hard_error.fallback_leg == SONNET46
    assert pr._ledger_entry_from_record({**pr._ledger_record(entries[0]), "retries": 1}) == entries[0]
    with pytest.raises(pr.PetriRunnerError, match="shed candidate"):
        replace(entries[0].hard_error, fallback_leg=None)       # a 429 is a hard error ONLY as a fallback leg's own failure
    assert pr.auditor_refusals_from_ledger(entries) == []
    # without the marker the same 429 event is an exhausted ladder (the ordinary shed path) — the marker is what decides
    plain = dict(smp, error={"message": "ModelGenerateError('x\n\nRateLimitError(\'Error code: 429 - rate limited\')')"})
    assert pr.ladder_exhaustion(plain)[:2] == (True, 429)


def test_a_judge_refusal_that_survived_the_client_side_fallback_is_filed_under_the_fallback_judge_and_never_re_rolled_again(world):
    """Seen live on a drive (2026-10-01 12:13Z): opus-5 refused, the client-side wrapper re-issued to opus-4-8, which
    refused too; the sample carried BOTH judges' events and the converter's single-other-model rule filed the refusal under opus-5
    -- read as unchanged, re-rollable forever. Now it is the fallback's: filed under opus-4-8, so rerollable_judge_refusals leaves
    it standing. The server-side one-event shape and the default-only shape keep their readings. Standing for the overflow pass
    is not final -- under a later chain whose TERMINAL leg never judged the sample, the judge-only re-score
    (rescorable_judge_refusals / petri_rescore) selects it, over the existing transcript, recorded under `rescored`."""
    from test_petri_judge_refusal import refused_sample
    from test_petri_store_shape import BEHAVIORS, convert
    spec = _amended(world.spec)
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    # client-side: the default's refusal event, the fallback's refusal event, twice over (petri's judge retries)
    two = refused_sample(SIDS[0]); two["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8")] * 2
    assert pr.refusing_judge_model(two, spec) == OPUS48
    _rows, entries = convert(world, cell, b, [two, sample(SIDS[1], full_dims(b))], spec=spec)
    (rec,) = pr.judge_refusals_from_ledger(entries)
    assert rec.judge_model == OPUS48 and OPUS48 in rec.reason
    standing, rerolled = pr.rerollable_judge_refusals([rec], spec, [])
    assert standing == [rec] and rerolled == []                                  # survived the fallback: never re-rolled again
    # Grok-fix-3: the overflow pass still never relaunches it; under the chain with the dated Grok terminal leg the judge-only re-score
    # selects it (the terminal leg never judged this sample), and under [opus-4-8] alone it does not (nothing new to try)
    grok_spec = replace(spec, judge_refusal_fallback_models=[OPUS48, GROK46])
    assert pr.rescorable_judge_refusals([rec], grok_spec, lambda r: ["claude-opus-5", "claude-opus-4-8"], [])[1] == [rec]
    assert pr.rescorable_judge_refusals([rec], spec, lambda r: ["claude-opus-5", "claude-opus-4-8"], [])[1] == []
    # server-side shape: one event, requested opus-5, served by the fallback
    one = refused_sample(SIDS[0]); one["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-4-8")]
    assert pr.refusing_judge_model(one, spec) == OPUS48
    # default only (no fallback reached): the manifest's judge, re-rollable
    only = refused_sample(SIDS[0]); only["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5")]
    assert pr.refusing_judge_model(only, spec) == spec.judge_model
    _rows, e2 = convert(world, cell, b, [only, sample(SIDS[1], full_dims(b))], spec=spec)
    (rec2,) = pr.judge_refusals_from_ledger(e2)
    assert pr.rerollable_judge_refusals([rec2], spec, [])[1] == [rec2]
    # no events at all, and a manifest without a declared judge fallback: the manifest's judge
    assert pr.refusing_judge_model(refused_sample(SIDS[0]), spec) == spec.judge_model
    assert not world.spec.judge_refusal_fallback_models and pr.refusing_judge_model(two, world.spec) == world.spec.judge_model


def test_with_several_declared_judge_fallbacks_the_terminal_names_the_last_attempted_one_not_the_lexicographically_last(world):
    """make_refusal_fallback_model tries the declared fallbacks in order and returns the
    LAST refusal, so the refusing judge is the last fallback-served event in the sample's chronology -- opus-4-5 when the manifest
    declares [opus-4-8, opus-4-5] and both refuse (a sort of the model ids would have said opus-4-8, falsifying the terminal
    judge, its reason and the attribution counts); the declared order reversed names opus-4-8; the refiled record stands."""
    from test_petri_judge_refusal import refused_sample
    from test_petri_store_shape import BEHAVIORS, convert
    OPUS45 = "anthropic/claude-opus-4-5"
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    spec = replace(world.spec, auditor_refusal_fallback_models=[SONNET46], judge_refusal_fallback_models=[OPUS48, OPUS45])
    both = refused_sample(SIDS[0])
    both["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8"), _event("judge", OPUS45, "claude-opus-4-5")]
    assert pr.refusing_judge_model(both, spec) == OPUS45                                             # the last attempted, not the sorted-last
    _rows, entries = convert(world, cell, b, [both, sample(SIDS[1], full_dims(b))], spec=spec)
    (rec,) = pr.judge_refusals_from_ledger(entries)
    assert rec.judge_model == OPUS45 and OPUS45 in rec.reason and OPUS48 not in rec.reason
    assert pr.rerollable_judge_refusals([rec], spec, [])[1] == []
    # the declared order reversed: the chronology follows the order tried, and the last tried is opus-4-8
    rev = replace(spec, judge_refusal_fallback_models=[OPUS45, OPUS48])
    both_rev = refused_sample(SIDS[0])
    both_rev["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS45, "claude-opus-4-5"), _event("judge", OPUS48, "claude-opus-4-8")]
    assert pr.refusing_judge_model(both_rev, rev) == OPUS48
    # petri's judge retried: two rounds of [default, fb1, fb2]; still the last attempted
    both["events"] = both["events"] * 2
    assert pr.refusing_judge_model(both, spec) == OPUS45
    # the first fallback served (scored under it) is never a refusal attribution question; a lone second-fallback event names it
    only_second = refused_sample(SIDS[0]); only_second["events"] = [_event("judge", OPUS45, "claude-opus-4-5")]
    assert pr.refusing_judge_model(only_second, spec) == OPUS45


def test_the_provenance_readers_count_a_client_side_hand_off_as_two_events_served_by_two_models():
    """Client-side, the refused default call and the fallback's call are two model events: the readers count the default
    once (it served a refusal) and the fallback once, both inside the amended contract, so the resume check admits
    the sample and the footing tally names the fallback-served request."""
    s = _served_sample("s1", auditor=("claude-sonnet-5", "claude-sonnet-4-6"), judge=("claude-opus-5",))
    assert pr.sample_role_serving_models(s) == {"auditor": {"claude-sonnet-5": 1, "claude-sonnet-4-6": 1}, "judge": {"claude-opus-5": 1}}
    allowed = pr.refusal_fallback_allowed(SimpleNamespace(auditor_model="anthropic/claude-sonnet-5", judge_model="anthropic/claude-opus-5",
                                                          auditor_refusal_fallback_models=[SONNET46], judge_refusal_fallback_models=[OPUS48]))
    served = pr.sample_role_serving_models(s)
    assert all(set(served[role]) <= allowed[role] for role in served)


# --- the provenance in rows, results and the footing line --------------------------------

def test_rows_carry_the_serving_models_only_under_an_amended_manifest(world):
    cell = world.cells["si_safety__psr_kl_all"]
    b = "self_preservation"
    rows, _ = convert(world, cell, b, [_served_sample(SIDS[0], auditor=("claude-sonnet-5", "claude-sonnet-4-6"), judge=("claude-opus-4-8",), behavior=b)], spec=_amended(world.spec))
    extras = [json.loads(r.extra_json) for r in rows]
    assert extras and all(e[pr.REFUSAL_FALLBACK_KEY] == {"auditor": {"claude-sonnet-5": 1, "claude-sonnet-4-6": 1}, "judge": {"claude-opus-4-8": 1}} for e in extras)
    rows_plain, _ = convert(world, cell, b, [_served_sample(SIDS[0], behavior=b)], spec=world.spec)
    assert all(pr.REFUSAL_FALLBACK_KEY not in json.loads(r.extra_json) for r in rows_plain)      # byte-identical rows without the fields


def test_the_tally_the_footing_and_the_line_sum_and_name_the_fallback_served_requests():
    served = [{"auditor": {"claude-sonnet-5": 14, "claude-sonnet-4-6": 1}, "judge": {"claude-opus-5": 1}},
              {"auditor": {"claude-sonnet-5": 15}, "judge": {"claude-opus-4-8": 1}}, "not a record", {"auditor": {"x": "n/a"}}]
    assert pr.refusal_fallback_tally(served) == {"auditor": {"claude-sonnet-5": 29, "claude-sonnet-4-6": 1}, "judge": {"claude-opus-5": 1, "claude-opus-4-8": 1}}
    behaviors = {"a": {"audited": 2, "n_items": 2, "auditor_refused": {}, "auditor_chain": {}, pr.REFUSAL_FALLBACK_KEY: served[0]},
                 "b": {"audited": 1, "n_items": 2, "auditor_refused": {}, "auditor_chain": {}, pr.REFUSAL_FALLBACK_KEY: served[1]}}
    footing = pr.audited_footing(behaviors)
    assert footing[pr.REFUSAL_FALLBACK_KEY] == pr.refusal_fallback_tally(served[:2])
    roles = {"auditor": "anthropic/claude-sonnet-5", "judge": "anthropic/claude-opus-5"}
    line = pr.audited_footing_line({"behaviors": behaviors, pr.ROLE_MODELS_KEY: roles})
    assert line.endswith(", refusal_fallback auditor claude-sonnet-4-6 1 / judge claude-opus-4-8 1")
    plain = {"behaviors": {"a": {"audited": 2, "n_items": 2, "auditor_refused": {}, "auditor_chain": {}}}}
    assert pr.REFUSAL_FALLBACK_KEY not in pr.audited_footing(plain["behaviors"]) and "refusal_fallback" not in pr.audited_footing_line(plain)
    # without the roles' defaults the serving models cannot be told apart: the line says so instead of guessing
    assert pr.refusal_fallback_footing({pr.REFUSAL_FALLBACK_KEY: {"auditor": {"claude-sonnet-5": 3, "claude-sonnet-4-6": 1}, "judge": {}}}) \
        == ", refusal_fallback: serving models recorded but the roles' defaults are not named (no role_models): unreported"
    assert pr.refusal_fallback_footing({pr.REFUSAL_FALLBACK_KEY: {"auditor": {"claude-sonnet-5": 3}, "judge": {}}}, roles) == ""


def _petri_run():
    """scripts/petri_run.py as a module (the sibling tests' way: on sys.path, so its dataclasses resolve their module)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import petri_run
    return petri_run


def test_the_registry_judgment_carries_the_judge_refused_count_and_the_fallback_notes():
    prun = _petri_run()
    served = {"auditor": {"claude-sonnet-5": 30, "claude-sonnet-4-6": 2}, "judge": {"claude-opus-5": 2, "claude-opus-4-8": 1}}
    items = [{"behavior": "b", "scenario_id": "s", "epoch": 1, "reason": "r", "judge_model": "anthropic/claude-opus-5"}] * 2
    analysis = {"by_cell": {"none": {"audited_footing": {"judge_refused": {"total": 2}, pr.REFUSAL_FALLBACK_KEY: served},
                                     "judge_refused": {"total": 2, "items": items}},                       # the first-class term, non-zero
                            "prompted__safety": {"audited_footing": {"judge_refused": {"total": 0}, pr.REFUSAL_FALLBACK_KEY: served}}}}
    # the notes helper reads the row's eval_id (petri_pooled -> the cell total, petri_<b> -> that
    # behaviour's count; a row without one refuses by name -- run.eval_id is NOT NULL, so none exists) and translates the TrainTest
    # object key to the harness id (an id the mapping cannot express refuses by name): the rows carry both, as every registry row does
    n, notes = prun.petri_judgment_notes(analysis, {"producer_object_key_legacy": "t:none", "eval_id": "petri_pooled"})
    assert n == 2 and notes == {"judge_refused": {"total": 2, "items": items}, pr.REFUSAL_FALLBACK_KEY: served}
    assert prun.petri_judgment_notes(analysis, {"producer_object_key_legacy": "t:prompted__safety_refusal", "eval_id": "petri_pooled"}) == (0, {"judge_refused": {"total": 0}, pr.REFUSAL_FALLBACK_KEY: served})
    # a run the analysis does not name -- a cell the adapter translates (antidote, Class B) with no entry here -- gets (None, {})
    assert prun.petri_judgment_notes(analysis, {"producer_object_key_legacy": "t:antidote", "eval_id": "petri_pooled"}) == (None, {})


# --- the generator ------------------------------------------------------------------

def test_the_generator_renders_the_two_blocks_for_the_s1_view_and_refuses_a_mistyped_chain():
    g = _generator()
    roster = _front_door_baseline_roster(g)
    text = g.render_manifest(roster, FRONT_DOOR, "battery", roster_path=g.ROSTER)
    assert f"audit: petri_full22_{FRONT_DOOR}_battery" in text
    # the printed block declares neither knob and the committed roster carries no chain: the two blocks below are the pass-through of
    # a chain this test declares, never a tree fact
    import copy as _copy, yaml as _yaml
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22")); sys.path.insert(0, str(REPO / "reproduce" / "lib"))
    import f22_roster as _FR, models as _MT
    printed = _yaml.load("targets:\n" + _MT.roster_block(FRONT_DOOR, ["battery"]), Loader=_FR._StrictRosterLoader)["targets"][FRONT_DOOR]["views"]["battery"]
    assert g.PETRI_AUDITOR_REFUSAL_FALLBACK_KEY not in printed and g.PETRI_JUDGE_REFUSAL_FALLBACK_KEY not in printed
    assert "refusal_fallback" not in (REPO / "configs" / "sdfbase_full22_roster.yaml").read_text(encoding="utf-8")
    bare = _copy.deepcopy(roster)
    for k in (g.PETRI_AUDITOR_REFUSAL_FALLBACK_KEY, g.PETRI_JUDGE_REFUSAL_FALLBACK_KEY):
        del bare["targets"][FRONT_DOOR]["views"]["battery"][k]
    assert "refusal_fallback" not in g.render_manifest(bare, FRONT_DOOR, "battery", roster_path=g.ROSTER)
    assert "\n  auditor_refusal_fallback_models:  # the Anthropic API's PER-REQUEST refusal fallback" in text and "\n    - anthropic/claude-sonnet-4-6\n" in text
    assert "\n  judge_refusal_fallback_models:  # the Anthropic API's PER-REQUEST refusal fallback" in text and "\n    - anthropic/claude-opus-4-8\n" in text
    assert text.index("judge_model: anthropic/claude-opus-5") < text.index("auditor_refusal_fallback_models") < text.index("scenarios_model:")
    other = g.render_manifest(roster, "qwen35_4b", "main", roster_path=g.ROSTER)    # a view without the knobs renders no block, byte for byte
    assert "refusal_fallback" not in other
    import copy
    assert "\n    - openrouter/x-ai/grok-4.6-20260810\n" in text and g._REFUSAL_FALLBACK_TERMINAL_TAIL in text     # The DATED terminal Grok leg renders
    bad0 = copy.deepcopy(roster); bad0["targets"][FRONT_DOOR]["views"]["battery"][g.PETRI_JUDGE_REFUSAL_FALLBACK_KEY] = [OPUS48, "openrouter/x-ai/grok-4.6"]
    with pytest.raises(SystemExit, match="IMMUTABLE"):                             # A mutable alias as the terminal leg refuses
        g.render_manifest(bad0, FRONT_DOOR, "battery", roster_path=g.ROSTER)
    bad = copy.deepcopy(roster); bad["targets"][FRONT_DOOR]["views"]["battery"][g.PETRI_JUDGE_REFUSAL_FALLBACK_KEY] = ["openrouter/x-ai/grok-4.6", OPUS48]
    with pytest.raises(SystemExit, match="TERMINAL"):                              # a non-Claude leg BEFORE a Claude one
        g.render_manifest(bad, FRONT_DOOR, "battery", roster_path=g.ROSTER)
    bad2 = copy.deepcopy(roster); bad2["targets"][FRONT_DOOR]["views"]["battery"][g.PETRI_AUDITOR_REFUSAL_FALLBACK_KEY] = [SONNET46, "openrouter/x-ai/grok-4.6"]
    with pytest.raises(SystemExit, match="first-party"):                           # the AUDITOR key stays first-party
        g.render_manifest(bad2, FRONT_DOOR, "battery", roster_path=g.ROSTER)


# --- X' --------------------------------

def test_the_rerun_window_scale_knob_moves_both_ends_and_refuses_a_misspelt_factor(monkeypatch):
    """PETRI_RERUN_WINDOW_SCALE=3 -> 49,152 x 8 (the smallest PRODUCTION_WINDOWS rung above the standard window, the
    certified product kept); unset -> the default doubling; anything but an int >= 2 refuses by name (never a silent fallback
    to a window the evidence rejected)."""
    monkeypatch.delenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, raising=False)
    assert pr.rerun_window_scale() == pr.RERUN_WINDOW_SCALE == 2
    assert pr.rerun_serving_geometry() == {"max_model_len": 32768, "max_num_seqs": 12}
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3")
    assert pr.rerun_window_scale() == 3
    geo = pr.rerun_serving_geometry()
    assert geo == {"max_model_len": 49152, "max_num_seqs": 8} and 49152 in __import__("sisteer.capeval.serving.standup_manifest", fromlist=["x"]).PRODUCTION_WINDOWS
    assert geo["max_model_len"] * geo["max_num_seqs"] == 24 * 16384
    for bad in ("1", "x", "2.5", "-3"):
        monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, bad)
        with pytest.raises(pr.PetriRunnerError, match="must be an integer >= 2"):
            pr.rerun_window_scale()


def test_the_footing_line_names_only_the_fallback_served_calls_given_the_roles_defaults():
    """A production-shaped results summary (both roles served by the default AND the fallback) must
    name only the fallback-served calls; a role served by the fallback alone is reported; without the roles' defaults the line
    says the counts are unreported rather than guessing."""
    behaviors = {"a": {"audited": 10, "n_items": 10, "auditor_refused": {}, "auditor_chain": {},
                       pr.REFUSAL_FALLBACK_KEY: {"auditor": {"claude-sonnet-5": 148, "claude-sonnet-4-6": 2}, "judge": {"claude-opus-5": 9, "claude-opus-4-8": 1}}},
                 "b": {"audited": 10, "n_items": 10, "auditor_refused": {}, "auditor_chain": {},
                       pr.REFUSAL_FALLBACK_KEY: {"auditor": {"claude-sonnet-5": 150}, "judge": {"claude-opus-4-8": 10}}}}
    summary = {"behaviors": behaviors, **pr.audited_footing(behaviors), pr.ROLE_MODELS_KEY: {"auditor": "anthropic/claude-sonnet-5", "judge": "anthropic/claude-opus-5"}}
    assert summary[pr.REFUSAL_FALLBACK_KEY] == {"auditor": {"claude-sonnet-5": 298, "claude-sonnet-4-6": 2}, "judge": {"claude-opus-5": 9, "claude-opus-4-8": 11}}
    line = pr.audited_footing_line(summary)
    assert line.endswith(", refusal_fallback auditor claude-sonnet-4-6 2 / judge claude-opus-4-8 11"), line
    assert "claude-sonnet-5" not in line.split("refusal_fallback")[1] and "claude-opus-5" not in line.split("refusal_fallback")[1]
    # the defaults may also be handed in explicitly (the runner passes the manifest's), and win over the summary's
    assert pr.audited_footing_line({k: v for k, v in summary.items() if k != pr.ROLE_MODELS_KEY}, {"auditor": "anthropic/claude-sonnet-5", "judge": "anthropic/claude-opus-5"}) == line
    only_fallback = {"behaviors": {"b": behaviors["b"]}, **pr.audited_footing({"b": behaviors["b"]}), pr.ROLE_MODELS_KEY: summary[pr.ROLE_MODELS_KEY]}
    assert pr.audited_footing_line(only_fallback).endswith(", refusal_fallback judge claude-opus-4-8 10")
    assert pr.refusal_fallback_footing({k: v for k, v in summary.items() if k != pr.ROLE_MODELS_KEY}) == \
        ", refusal_fallback: serving models recorded but the roles' defaults are not named (no role_models): unreported"
    assert pr.role_models_of(SimpleNamespace(auditor_model="anthropic/claude-sonnet-5", judge_model="anthropic/claude-opus-5")) == summary[pr.ROLE_MODELS_KEY]


def _jr(behavior, sid, judge="anthropic/claude-opus-5"):
    return pr.JudgeRefusal(behavior=behavior, scenario_id=sid, epoch=1, dimensions=(behavior, "eval_awareness", "scenario_realism"),
                           reason=f"judge refused to score ({judge}): x", judge_model=judge)


def test_the_overflow_pass_rerolls_default_judge_refusals_once_under_a_declared_fallback(world, tmp_path):
    """With a judge refusal fallback declared, prior terminal refusals by the DEFAULT judge leave the standing set
    (relaunched from turn 1, judged under the fallback) and move to the sidecar's `rerolled` list with what re-rolled them;
    a refusal by the fallback judge stands; a record re-rolled once is never re-rolled again; without a declared fallback
    nothing moves. The sidecar keeps the re-rolled records across rewrites and the reader ignores them."""
    plain = world.spec; amended = _amended(plain)
    prior = [_jr("self_preservation", SIDS[0]), _jr("defer_to_users", SIDS[1], judge="anthropic/claude-opus-4-8")]
    assert pr.rerollable_judge_refusals(prior, plain) == (prior, [])                          # no fallback declared: nothing moves
    standing, rerolled = pr.rerollable_judge_refusals(prior, amended)
    assert [r.scenario_id for r in rerolled] == [SIDS[0]] and [r.scenario_id for r in standing] == [SIDS[1]]
    already = [{"behavior": "self_preservation", "scenario_id": SIDS[0], "epoch": 1}]
    assert pr.rerollable_judge_refusals(prior, amended, already) == (prior, [])              # re-rolled once: never again
    # the sidecar round trip
    from sisteer.capeval import store as cs
    cell = world.cells["si_safety__psr_kl_all"]
    store = SimpleNamespace(cell_dir=tmp_path / "cell")
    records = [{"behavior": "self_preservation", "scenario_id": SIDS[0], "epoch": 1, "rerolled_by": "overflow-rerun pass", "rerun_window": 49152}]
    payload = pr.write_judge_refused(store, cell, standing, rerolled=records)
    assert payload[pr.JUDGE_REROLLED_KEY] == records and [r["scenario_id"] for r in payload[pr.JUDGE_REFUSED_OUTCOME]] == [SIDS[1]]
    assert [r.scenario_id for r in pr.read_judge_refused(store)] == [SIDS[1]] and pr.read_judge_rerolled(store) == records
    pr.write_judge_refused(store, cell, standing)                                            # a rewrite without the argument keeps them
    assert pr.read_judge_rerolled(store) == records
    assert pr.write_judge_refused(store, cell, standing, rerolled=[]).get(pr.JUDGE_REROLLED_KEY) is None


def test_arming_the_overflow_pass_records_the_relaunch_set_overflowed_and_refused(world, tmp_path, monkeypatch):
    """petri_rerun_relaunch.json names the window served, the factor, the overflowed scenario ids and the judge-refused
    ids re-rolled under the fallback, per behaviour; the footing collector hands it on as `rerun_pass`."""
    prun = _petri_run()
    cell_dir = tmp_path / "cells" / "deadbeef"; cell_dir.mkdir(parents=True)
    ledger = {"unresolved": [{"behavior": "self_preservation", "scenario_id": SIDS[0], "epoch": 1, "dimension": d,
                              "reason": pr.OVERFLOW_REASON_PREFIX + " the serving window bound mid-audit"} for d in ("self_preservation", "eval_awareness", "scenario_realism")]
                             + [{"behavior": "defer_to_users", "scenario_id": SIDS[1], "epoch": 1, "dimension": "defer_to_users", "reason": "something else"}]}
    (cell_dir / prun.FAILURE_LEDGER).write_text(json.dumps(ledger))
    cell = world.cells["si_safety__psr_kl_all"]
    pr.write_judge_refused(SimpleNamespace(cell_dir=cell_dir), cell, [_jr("instruction_evasion", SIDS[1])])
    # arming sets PETRI_OVERFLOW_RERUN=1 in os.environ directly: register the variable with monkeypatch FIRST so its teardown removes
    # it (a bare delenv of an absent name records nothing, and the "1" would leak into every later test of the process)
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    armed = prun._arm_overflow_rerun(cell_dir, "si_safety__psr_kl_all", _amended(world.spec))
    assert len(armed) == 3 and os.environ.get(pr.PETRI_OVERFLOW_RERUN_ENV) == "1"
    rec = json.loads((cell_dir / prun.RERUN_RELAUNCH_RECORD).read_text())
    assert rec["window"] == 49152 and rec["max_num_seqs"] == 8 and rec["scale"] == 3
    assert rec["overflowed"] == [["self_preservation", SIDS[0]]] and rec["refused"] == [["instruction_evasion", SIDS[1]]]
    assert rec["judge_refusal_fallback_models"] == [OPUS48] and rec["auditor_refusal_fallback_models"] == [SONNET46]
    assert prun._rerun_pass_of(SimpleNamespace(path=cell_dir)) == {"rerun_pass": rec} and prun._rerun_pass_of(SimpleNamespace(path=tmp_path)) == {}
    # without a declared judge fallback the refused set is empty and nothing is re-rolled
    rec2_dir = tmp_path / "cells" / "cafe"; rec2_dir.mkdir(); (rec2_dir / prun.FAILURE_LEDGER).write_text(json.dumps(ledger))
    pr.write_judge_refused(SimpleNamespace(cell_dir=rec2_dir), cell, [_jr("instruction_evasion", SIDS[1])])
    prun._arm_overflow_rerun(rec2_dir, "x", world.spec)
    assert json.loads((rec2_dir / prun.RERUN_RELAUNCH_RECORD).read_text())["refused"] == []


def test_the_registry_judgment_notes_read_the_real_analysis_shape():
    """Through the REAL petri_stats.analyze: the per-cell counts live under
    by_cell[cell]['audited_footing'] -- a zero-refusal arm gets n_judge_refusals=0 and both roles' serving-model counts; an arm
    with judge refusals keeps the first-class non-zero term (items with reasons) as its notes."""
    from test_petri_stats import BEHAVIORS3, LENS, RUN_KEY, _analyze, _cell_footing, _enumeration_payload, _provenance_for, build_world
    from test_petri_judge_refusal import _drop_items, _footing_with_judge
    from sisteer.capeval.petri import defense_cells as dc
    prun = _petri_run()
    method = dc.PROMPTABLE_CLASS_A_METHODS[0]
    cell = f"si_constitution__{method}"
    served = {"auditor": {"claude-sonnet-5": 268, "claude-sonnet-4-6": 2}, "judge": {"claude-opus-5": 17, "claude-opus-4-8": 1}}
    roles = {"auditor": "anthropic/claude-sonnet-5", "judge": "anthropic/claude-opus-5"}
    rerun = {"window": 49152, "scale": 3, "overflowed": [["self_preservation", "scenario_alpha"]], "refused": []}
    # (a) a complete, zero-refusal arm whose footing carries the counts
    frame = build_world()
    footing = _cell_footing({b: (6, 6, 0) for b in BEHAVIORS3}, run_key=RUN_KEY)
    footing.update({"judge_refused": {"total": 0}, pr.REFUSAL_FALLBACK_KEY: served, pr.ROLE_MODELS_KEY: roles, "rerun_pass": rerun})
    provenance = _provenance_for(frame); provenance["cells"][cell] = _enumeration_payload(LENS)
    analysis = _analyze(frame, cell_footing={cell: footing}, provenance=provenance)
    entry = analysis["by_cell"][cell]
    assert entry["audited_footing"][pr.REFUSAL_FALLBACK_KEY] == served and "judge_refused" not in entry   # zero adds no first-class term
    n, notes = prun.petri_judgment_notes(analysis, {"producer_object_key_legacy": f"t:{cell}", "eval_id": "petri_pooled"})   # The pooled run's notes = the cell's
    assert n == 0 and notes["judge_refused"] == {"total": 0} and notes[pr.REFUSAL_FALLBACK_KEY] == served
    assert notes[pr.ROLE_MODELS_KEY] == roles and notes["rerun_pass"] == rerun
    # (b) an arm with a judge refusal: the first-class term (items with reasons) is what the notes carry, and the count agrees
    b = "self_preservation"; sid = "s5"                                     # the synthetic stats world's items are s0..s5
    frame2 = _drop_items(build_world(), cell, b, [sid])
    footing2 = _footing_with_judge({bb: (6, 6 if bb != b else 5, 0) for bb in BEHAVIORS3}, {b: [sid]}, run_key=RUN_KEY)
    footing2.update({pr.REFUSAL_FALLBACK_KEY: served, pr.ROLE_MODELS_KEY: roles})
    provenance2 = _provenance_for(frame2); provenance2["cells"][cell] = _enumeration_payload(LENS)
    analysis2 = _analyze(frame2, cell_footing={cell: footing2}, provenance=provenance2)
    n2, notes2 = prun.petri_judgment_notes(analysis2, {"producer_object_key_legacy": f"t:{cell}", "eval_id": "petri_pooled"})
    assert n2 == 1 and notes2["judge_refused"]["total"] == 1 and notes2["judge_refused"]["items"][0]["scenario_id"] == sid
    assert notes2[pr.REFUSAL_FALLBACK_KEY] == served and "rerun_pass" not in notes2
    # a run the analysis does not name -- a translatable cell (antidote) this analysis has no entry for -- gets (None, {})
    trimmed2 = {**analysis2, "by_cell": {k: v for k, v in analysis2["by_cell"].items() if k != "antidote"}}
    assert prun.petri_judgment_notes(trimmed2, {"producer_object_key_legacy": "t:antidote", "eval_id": "petri_pooled"}) == (None, {})
    # a per-behaviour run reads ITS behaviour's refusal count (the pooled run the cell's, above): the refused
    # behaviour's run carries the one refusal with its item, another behaviour's run carries none
    n_b, notes_b = prun.petri_judgment_notes(analysis2, {"producer_object_key_legacy": f"t:{cell}", "eval_id": f"petri_{b}"})
    assert n_b == 1 and notes_b["judge_refused"]["total"] == 1 and notes_b["judge_refused"]["items"][0]["behavior"] == b
    n_o, notes_o = prun.petri_judgment_notes(analysis2, {"producer_object_key_legacy": f"t:{cell}", "eval_id": "petri_defer_to_users"})
    assert n_o == 0 and notes_o["judge_refused"]["total"] == 0 and notes_o[pr.REFUSAL_FALLBACK_KEY] == served


# --- X'': the record IS the task list; re-roll stamped from post-launch evidence -------------

def _seed_store(world, cell, tmp_path, *, overflow=(), api_error=(), judge_refused=()):
    """A store the way a FIRST pass leaves a cell: one real ``_run`` with a seeding driver whose logs carry, per (behaviour, scenario),
    a scored sample, a context-OVERFLOW casualty (the target-failure sentinel + a context-length marker in a message, the converter's
    normal surface), a sample-level API error (an unresolved entry of another class), or a judge REFUSAL by the default judge -- so
    manifest.json, rows.jsonl, the ledger and the judge-refused sidecar are the runner's own."""
    from sisteer.capeval import store as cs
    from test_petri_judge_refusal import refused_sample
    store = cs.CellStore(tmp_path / "sweep", pr.petri_cell_key(cell))
    over, api, ref = set(overflow), set(api_error), set(judge_refused)

    def seed(tasks, *, spec, cell, engine):
        logs = []
        for t in tasks:
            samples = []
            for sid in SIDS:
                key = (t.behavior_id, sid)
                if key in over:
                    smp = sample(sid, full_dims(t.behavior_id))
                    smp["messages"] = [{"role": "tool", "content": pr.TARGET_FAILURE_SENTINEL + " This model's maximum context length is 16384 tokens. However, you requested 1024 output tokens and your prompt contains at least 15361 input tokens"}]
                    samples.append(smp)
                elif key in api:
                    samples.append(sample(sid, None, error={"message": "APIStatusError 500 from the auditor pool"}))
                elif key in ref:
                    samples.append(refused_sample(sid))
                else:
                    samples.append(sample(sid, full_dims(t.behavior_id)))
            logs.append(make_log(world, cell, t.dir_key, samples, spec=spec))
        return logs
    first = _run_pass(world, cell, store, tmp_path, seed, world.spec)
    assert first["incomplete"] is True
    return store


def _overflow_casualty(sid, behavior):
    smp = sample(sid, full_dims(behavior))
    smp["messages"] = [{"role": "tool", "content": pr.TARGET_FAILURE_SENTINEL + " This model's maximum context length is 16384 tokens. However, you requested 1024 output tokens and your prompt contains at least 15361 input tokens"}]
    return smp


def _capturing_driver(world, cell, *, mode="score", overflow=(), refused=()):
    """A stub inspect driver that records the task list it received and answers as: ``score`` (every requested scenario scored),
    ``replay`` (a replayed resume: the SAME casualty shapes as before -- the overflow casualty and the default-judge refusal -- so
    nothing changes), or ``fallback_refuses`` (the re-rolled refusal is refused AGAIN, this time with judge events naming the
    fallback judge; everything else scored)."""
    from test_petri_judge_refusal import refused_sample
    seen = []

    def drive(tasks, *, spec, cell, engine):
        seen.append([(t.behavior_id, tuple(t.scenario_ids) if t.scenario_ids is not None else None) for t in tasks])
        logs = []
        for t in tasks:
            samples = []
            for sid in (t.scenario_ids or SIDS):
                key = (t.behavior_id, sid)
                if mode == "replay" and key in set(overflow):
                    samples.append(_overflow_casualty(sid, t.behavior_id))
                elif mode == "replay" and key in set(refused):
                    samples.append(refused_sample(sid))
                elif mode == "fallback_refuses" and key in set(refused):
                    smp = refused_sample(sid)
                    smp["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-4-8")]
                    samples.append(smp)
                else:
                    samples.append(sample(sid, full_dims(t.behavior_id)))
            logs.append(make_log(world, cell, t.dir_key, samples, spec=spec))
        return logs
    drive.seen = seen
    return drive


def _run_pass(world, cell, store, tmp_path, driver, spec):
    return pr._run(object(), spec, cell, store, seed=spec.bloom_options.seed, driver=driver, config_root=world.config_root,
                   bundle_dir=world.bundle.bundle_dir, work_root=tmp_path / "work", log_root=tmp_path / "logs", prompt_files=PROMPT_FILES)


def test_the_overflow_pass_launches_exactly_the_recorded_union_and_records_what_it_left(world, tmp_path, monkeypatch):
    """Against the REAL runner and its driver's task list: a cell with one overflow casualty, one scenario
    unresolved for another reason (an API error) and one default-judge refusal. The pass relaunches the overflowed and the re-rolled
    scenario and NOTHING else; the relaunch record names what launched (with its reason) and what it left unresolved."""
    prun = _petri_run()
    spec = _amended(world.spec)
    cell = world.cells["si_safety__psr_kl_all"]
    from test_petri_store_shape import BEHAVIORS
    b_over, b_api, b_ref = BEHAVIORS[0], BEHAVIORS[1], BEHAVIORS[1]            # the synthetic world's two behaviours x two scenarios
    store = _seed_store(world, cell, tmp_path, overflow=[(b_over, SIDS[0])], api_error=[(b_api, SIDS[0])], judge_refused=[(b_ref, SIDS[1])])
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    armed = prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)
    assert len(armed) == 3 and os.environ[pr.PETRI_OVERFLOW_RERUN_ENV] == "1"
    rec = pr.read_rerun_relaunch(store.cell_dir)
    assert rec["overflowed"] == [[b_over, SIDS[0]]] and rec["refused"] == [[b_ref, SIDS[1]]] and rec["window"] == 49152
    driver = _capturing_driver(world, cell)
    result = _run_pass(world, cell, store, tmp_path, driver, spec)
    (tasks,) = driver.seen
    assert sorted(tasks) == sorted([(b_over, (SIDS[0],)), (b_ref, (SIDS[1],))]), tasks   # the API-unresolved scenario (b_api, SIDS[0]) was NOT launched
    rec = pr.read_rerun_relaunch(store.cell_dir)
    assert sorted(rec["launched"]) == sorted([[b_ref, SIDS[1], "refused"], [b_over, SIDS[0], "overflowed"]]) and rec["left_unresolved"] == [[b_api, SIDS[0]]]
    assert result["incomplete"] is True and result["unscored"] == len(pr.required_metrics(b_api))   # the API scenario still stands in the ledger
    # the re-rolled refusal was SCORED under the pass: stamped from post-launch evidence, out of the standing set, with its outcome
    (stamp,) = pr.read_judge_rerolled(store)
    assert (stamp["behavior"], stamp["scenario_id"], stamp["outcome"], stamp["rerun_window"]) == (b_ref, SIDS[1], "scored", 49152)
    assert pr.read_judge_refused(store) == []
    assert pr.judge_refused_row_keys([_jr(b_ref, SIDS[1])], spec.bloom_options.epochs) <= store.completed_keys()
    prun._assert_overflow_rerun_progress(store.cell_dir, cell.cell_id, armed, pre_rerolled=0)   # real progress: nothing raised


def test_a_relaunch_that_changes_nothing_keeps_the_refusal_record_and_counts_as_no_progress(world, tmp_path, monkeypatch):
    """A replayed resume (the driver returns nothing) must leave the re-rolled item's standing record in
    place, stamp nothing, and the pass must refuse as zero-progress -- the pre-run record's `refused` list alone proves nothing."""
    prun = _petri_run()
    spec = _amended(world.spec)
    cell = world.cells["si_safety__psr_kl_all"]
    from test_petri_store_shape import BEHAVIORS
    b_over, b_ref = BEHAVIORS[0], BEHAVIORS[1]
    store = _seed_store(world, cell, tmp_path, overflow=[(b_over, SIDS[0])], judge_refused=[(b_ref, SIDS[1])])
    # an armed pass with no relaunch record refuses to guess which scenarios it may relaunch (the record is what the runner launches from)
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "1")
    with pytest.raises(pr.PetriRunnerError, match="no petri_rerun_relaunch.json names the pass's relaunch set"):
        _run_pass(world, cell, store, tmp_path, _capturing_driver(world, cell), spec)
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    armed = prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)
    pre = len(pr.read_judge_rerolled(store)); assert pre == 0
    driver = _capturing_driver(world, cell, mode="replay", overflow=[(b_over, SIDS[0])], refused=[(b_ref, SIDS[1])])
    result = _run_pass(world, cell, store, tmp_path, driver, spec)
    (tasks,) = driver.seen
    assert sorted(tasks) == sorted([(b_over, (SIDS[0],)), (b_ref, (SIDS[1],))])
    assert result["incomplete"] is True
    standing = pr.read_judge_refused(store)
    assert pr.read_judge_rerolled(store) == [] and [(r.scenario_id, r.judge_model) for r in standing] == [(SIDS[1], "anthropic/claude-opus-5")]   # unchanged: record kept
    with pytest.raises(prun.DriverError, match="resolved NONE"):
        prun._assert_overflow_rerun_progress(store.cell_dir, cell.cell_id, armed, pre_rerolled=pre)
    # the refusal refused AGAIN, by the FALLBACK judge: a change -- stamped refiled_terminal, the standing record now names the fallback
    # judge (never re-rolled again), and the pass counts as progress
    driver2 = _capturing_driver(world, cell, mode="fallback_refuses", refused=[(b_ref, SIDS[1])])
    _run_pass(world, cell, store, tmp_path, driver2, spec)
    (stamp,) = pr.read_judge_rerolled(store)
    assert (stamp["scenario_id"], stamp["outcome"]) == (SIDS[1], "refiled_terminal")
    assert [(r.scenario_id, r.judge_model) for r in pr.read_judge_refused(store)] == [(SIDS[1], "anthropic/claude-opus-4-8")]
    assert pr.rerollable_judge_refusals(pr.read_judge_refused(store), spec, pr.read_judge_rerolled(store))[1] == []
    prun._assert_overflow_rerun_progress(store.cell_dir, cell.cell_id, armed, pre_rerolled=pre)   # progress: nothing raised


# --- X''': a fully covered behaviour must still relaunch through a resume sub-namespace -----------------

def _stub_inspect_for_driver(monkeypatch, tmp_path):
    """The inspect / petri_bloom seams stubbed around the REAL _inspect_driver; returns the eval_set kwargs it receives."""
    calls = []
    inspect_pkg = types.ModuleType("inspect_ai")
    inspect_pkg.eval_set = lambda group, **kwargs: (calls.append(dict(kwargs)) or (True, []))
    inspect_pkg.task_with = lambda task, *, model_roles, metadata: task
    log_mod = types.ModuleType("inspect_ai.log"); log_mod.read_eval_log = lambda location: None
    model_mod = types.ModuleType("inspect_ai.model")

    class FakeGenerateConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def merge(self, other):
            return self
    model_mod.GenerateConfig = FakeGenerateConfig
    model_mod.get_model = lambda name, **kwargs: SimpleNamespace(name=name, config=kwargs.get("config"), _set_role=lambda role: None)
    bloom = types.ModuleType("petri_bloom"); bloom.bloom_audit = lambda **kwargs: object()
    for name, mod in (("inspect_ai", inspect_pkg), ("inspect_ai.log", log_mod), ("inspect_ai.model", model_mod), ("petri_bloom", bloom)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pr, "_require_inspect", lambda: inspect_pkg)
    monkeypatch.setattr(pr, "require_agent_loop_entry", lambda engine, cell: None)
    monkeypatch.setattr(pr, "eval_concurrency", lambda spec, engine: 4)
    monkeypatch.setattr(pr, "install_retry_sleep_cap", lambda: None)
    monkeypatch.setattr(pr, "eval_set_bounds", lambda: {})
    monkeypatch.setattr(af, "make_inspect_auditor_chain", lambda p, f, **kw: ("AUDITOR-CHAIN", p, tuple(f)))
    return calls


def test_the_real_inspect_driver_puts_an_explicit_full_subset_in_a_resume_namespace(world, tmp_path, monkeypatch):
    """Through the REAL _inspect_driver: a task whose scenario_ids tuple covers every canonical scenario still
    gets eval_set's `sample_id` and a `resume-<sha16(subset)>` log_dir (fresh requests), while a task with scenario_ids=None (a plain
    full launch) gets the top-level log_dir and no sample_id — the two shapes the overflow pass must tell apart."""
    calls = _stub_inspect_for_driver(monkeypatch, tmp_path)
    spec = replace(world.spec, auditor_fallback_models=[], judge_fallback_models=[])
    cell = _cell_under(spec, world)
    engine = SimpleNamespace(petri_target=lambda cell: SimpleNamespace(config=None))
    log_dir = tmp_path / "logs"
    full = pr.EvalTaskSpec(behavior_id="self_preservation", dir_key="self_preservation", behavior_dir=tmp_path, log_dir=log_dir, metadata={},
                           options={}, scenario_ids=tuple(SIDS))
    plain = pr.EvalTaskSpec(behavior_id="self_preservation", dir_key="self_preservation", behavior_dir=tmp_path, log_dir=log_dir, metadata={},
                            options={}, scenario_ids=None)
    pr._inspect_driver([full], spec=spec, cell=cell, engine=engine)
    pr._inspect_driver([plain], spec=spec, cell=cell, engine=engine)
    assert len(calls) == 2
    sub = cm.sha256_hex16(cm.canonical_json(list(SIDS)).encode("utf-8"))
    assert calls[0]["sample_id"] == list(SIDS) and calls[0]["log_dir"] == str(log_dir / f"resume-{sub}")
    assert "sample_id" not in calls[1] and calls[1]["log_dir"] == str(log_dir)


def test_the_overflow_pass_keeps_the_explicit_tuple_when_every_scenario_of_a_behaviour_is_relaunched(world, tmp_path, monkeypatch):
    """The runner's side of the same defect: under the armed pass a behaviour whose EVERY canonical scenario is in the relaunch set is
    handed to the driver as the explicit tuple (a resume sub-namespace, fresh requests at the wider window), never as None (the
    top-level log dir, where Inspect would replay the casualties completed at 16,384); a plain run still collapses a full set to None."""
    from test_petri_store_shape import BEHAVIORS
    prun = _petri_run()
    spec = _amended(world.spec)
    cell = world.cells["si_safety__psr_kl_all"]
    b_all, b_part = BEHAVIORS[0], BEHAVIORS[1]
    store = _seed_store(world, cell, tmp_path, overflow=[(b_all, SIDS[0]), (b_all, SIDS[1]), (b_part, SIDS[0])])   # b_all fully overflowed
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)
    driver = _capturing_driver(world, cell)
    _run_pass(world, cell, store, tmp_path, driver, spec)
    (tasks,) = driver.seen
    assert dict(tasks) == {b_all: tuple(sorted(SIDS)), b_part: (SIDS[0],)}, tasks          # explicit, never None
    rec = pr.read_rerun_relaunch(store.cell_dir)
    assert sorted(rec["launched"]) == sorted([[b_all, SIDS[0], "overflowed"], [b_all, SIDS[1], "overflowed"], [b_part, SIDS[0], "overflowed"]])
    # the same cell driven as a PLAIN resume (not armed) collapses a fully missing behaviour to None, as it always did
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    store2 = _seed_store(world, cell, tmp_path / "plain", overflow=[(b_all, SIDS[0]), (b_all, SIDS[1])])
    driver2 = _capturing_driver(world, cell)
    _run_pass(world, cell, store2, tmp_path / "plain", driver2, world.spec)
    (tasks2,) = driver2.seen
    assert dict(tasks2) == {b_all: None}, tasks2


# --- X'''': the armed pass's resume namespace is keyed by the served window AND the subset ----------------

def test_the_armed_pass_resumes_in_a_window_keyed_namespace_never_the_ordinary_subset_directory(world, tmp_path, monkeypatch):
    """Through the REAL _inspect_driver with a stubbed inspect_ai.eval_set: an ordinary filtered resume keeps its subset-only directory
    (`resume-<sha16(subset)>`, byte-identical to every existing one); the armed pass of the same subset gets `resume-w<window>-<sha16>`,
    never the ordinary directory even when it is already populated with samples completed at the standard window (the collision
    Inspect would replay); a pass at another window gets another directory; the same window twice gets the same one (stable across a
    same-window crash/restart)."""
    calls = _stub_inspect_for_driver(monkeypatch, tmp_path)
    spec = replace(world.spec, auditor_fallback_models=[], judge_fallback_models=[])
    cell = _cell_under(spec, world)
    engine = SimpleNamespace(petri_target=lambda cell: SimpleNamespace(config=None))
    log_dir = tmp_path / "logs"
    task = pr.EvalTaskSpec(behavior_id="self_preservation", dir_key="self_preservation", behavior_dir=tmp_path, log_dir=log_dir, metadata={},
                           options={}, scenario_ids=tuple(SIDS))
    monkeypatch.delenv(pr.PETRI_OVERFLOW_RERUN_ENV, raising=False); monkeypatch.delenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, raising=False)
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)                    # an ordinary filtered resume
    plain = Path(calls[-1]["log_dir"])
    assert plain == log_dir / f"resume-{cm.sha256_hex16(cm.canonical_json(list(SIDS)).encode('utf-8'))}" and plain.name == pr.resume_namespace(SIDS)
    plain.mkdir(parents=True); (plain / "2026-09-30T20-27-00_bloom-audit_old.eval").write_bytes(b"completed at 16384")   # populated
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "1"); monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3")
    monkeypatch.setenv(pr.PETRI_SERVED_MAX_MODEL_LEN_ENV, "49152")                    # the proxy attests the same window (X5)
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)                    # the armed pass at 49,152
    armed = Path(calls[-1]["log_dir"])
    assert armed != plain and armed.parent == log_dir and armed.name.startswith("resume-w49152-") and calls[-1]["sample_id"] == list(SIDS)
    assert armed.name == pr.resume_namespace(SIDS) and not armed.exists()               # fresh: nothing to replay
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)                    # the same window again: the same directory
    assert Path(calls[-1]["log_dir"]) == armed
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "2"); monkeypatch.setenv(pr.PETRI_SERVED_MAX_MODEL_LEN_ENV, "32768")
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)                    # another (attested) window: another directory
    other = Path(calls[-1]["log_dir"])
    assert other.name.startswith("resume-w32768-") and other not in (armed, plain)
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)                    # disarmed: the ordinary directory again
    assert Path(calls[-1]["log_dir"]) == plain


# --- X5: the namespace's window is the one the proxy ATTESTS, held equal to the requested one --------------

def test_the_armed_pass_keys_by_the_attested_window_and_refuses_a_proxy_drive_scale_mismatch(world, tmp_path, monkeypatch):
    """Through the REAL _inspect_driver with a stubbed inspect_ai.eval_set: the proxy attests 49,152 (cells_served.json
    attested_max_model_len, handed in as PETRI_SERVED_MAX_MODEL_LEN); a drive requesting scale 3 gets `resume-w49152-…` (the attested
    window) and eval_set runs; a drive requesting scale 2 (32,768) is REFUSED before any eval_set call, naming both windows; an armed
    pass with no attestation is refused; an ordinary (unarmed) resume is untouched by either variable."""
    calls = _stub_inspect_for_driver(monkeypatch, tmp_path)
    spec = replace(world.spec, auditor_fallback_models=[], judge_fallback_models=[])
    cell = _cell_under(spec, world)
    engine = SimpleNamespace(petri_target=lambda cell: SimpleNamespace(config=None))
    log_dir = tmp_path / "logs"
    task = pr.EvalTaskSpec(behavior_id="self_preservation", dir_key="self_preservation", behavior_dir=tmp_path, log_dir=log_dir, metadata={},
                           options={}, scenario_ids=tuple(SIDS))
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "1"); monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3")
    monkeypatch.setenv(pr.PETRI_SERVED_MAX_MODEL_LEN_ENV, "49152")
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)
    assert len(calls) == 1 and Path(calls[0]["log_dir"]).name.startswith("resume-w49152-") and calls[0]["sample_id"] == list(SIDS)
    assert pr.served_window() == 49152 and pr.rerun_geometry_or_refuse() == 49152
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "2")                      # the drive asks 32,768 against a 49,152 proxy
    with pytest.raises(pr.PetriRunnerError, match=r"requests max_model_len=32768 .* proxy attests max_model_len=49152"):
        pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)
    assert len(calls) == 1                                                         # refused BEFORE eval_set
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.delenv(pr.PETRI_SERVED_MAX_MODEL_LEN_ENV)
    with pytest.raises(pr.PetriRunnerError, match="served window is not known"):
        pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)
    assert len(calls) == 1
    monkeypatch.setenv(pr.PETRI_SERVED_MAX_MODEL_LEN_ENV, "many")
    with pytest.raises(pr.PetriRunnerError, match="must be the standup's attested max_model_len"):
        pr.served_window()
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0"); monkeypatch.setenv(pr.PETRI_SERVED_MAX_MODEL_LEN_ENV, "49152")
    pr._inspect_driver([task], spec=spec, cell=cell, engine=engine)              # unarmed: the ordinary subset directory, no geometry check
    assert Path(calls[-1]["log_dir"]) == log_dir / f"resume-{cm.sha256_hex16(cm.canonical_json(list(SIDS)).encode('utf-8'))}"


def test_arming_reads_the_attested_window_off_the_standup_evidence_and_refuses_without_it(tmp_path):
    """petri_run.py's armed pass hands the runner the window the proxy ATTESTS (cells_served.json attested_max_model_len, the file the
    drive pulls beside the endpoint and passes as --verify-standup); no file, no attestation, or a non-integer refuses."""
    prun = _petri_run()
    p = tmp_path / "cells_served.json"
    p.write_text(json.dumps({"engine": "lora_native", "stage": "final", "attested_max_model_len": 49152, "attested_max_num_seqs": 4}))
    assert prun.served_window_from_standup(str(p)) == 49152
    for bad in (None, ""):
        with pytest.raises(prun.DriverError, match="needs --verify-standup"):
            prun.served_window_from_standup(bad)
    p.write_text(json.dumps({"engine": "lora_native", "attested_max_model_len": None}))
    with pytest.raises(prun.DriverError, match="no attested_max_model_len"):
        prun.served_window_from_standup(str(p))
    p.write_text("{not json")
    with pytest.raises(prun.DriverError, match="cannot read the standup evidence"):
        prun.served_window_from_standup(str(p))
    with pytest.raises(prun.DriverError, match="cannot read the standup evidence"):
        prun.served_window_from_standup(str(tmp_path / "absent.json"))


def test_arming_reads_the_attested_window_off_the_endpoint_record_on_the_lens_path_and_holds_two_records_equal(tmp_path):
    """X6: the lens drive pulls no cells_served.json, so the armed pass reads the proxy's attestation from the pulled
    endpoint.json (--served-endpoint; common.sh writes attested_max_model_len for either engine). Both records given must agree; neither
    refuses naming both flags; an endpoint attesting no window refuses."""
    prun = _petri_run()
    ep = tmp_path / "endpoint.json"; cs = tmp_path / "cells_served.json"
    ep.write_text(json.dumps({"node": "n", "slurm_job_id": "1", "mode": "petri", "lens_reduction_mode": "stacked", "attested_max_model_len": 49152}))
    assert prun.served_window_from_endpoint(str(ep)) == 49152
    assert prun.served_window_from_evidence(None, str(ep)) == 49152                       # the lens drive's shape
    cs.write_text(json.dumps({"engine": "lora_native", "stage": "final", "attested_max_model_len": 49152}))
    assert prun.served_window_from_evidence(str(cs), str(ep)) == 49152                    # the native drive, both records
    assert prun.served_window_from_evidence(str(cs), None) == 49152
    cs.write_text(json.dumps({"engine": "lora_native", "stage": "final", "attested_max_model_len": 32768}))
    with pytest.raises(prun.DriverError, match=r"attestations disagree .*--verify-standup attests max_model_len=32768, --served-endpoint attests max_model_len=49152"):
        prun.served_window_from_evidence(str(cs), str(ep))
    with pytest.raises(prun.DriverError, match=r"needs the proxy's attestation: --verify-standup .* or --served-endpoint"):
        prun.served_window_from_evidence(None, None)
    ep.write_text(json.dumps({"node": "n", "mode": "petri", "lens_reduction_mode": "stacked"}))
    with pytest.raises(prun.DriverError, match="carries no attested_max_model_len"):
        prun.served_window_from_evidence(None, str(ep))
    ep.write_text("{not json")
    with pytest.raises(prun.DriverError, match="cannot read the endpoint record"):
        prun.served_window_from_endpoint(str(ep))
    # the flag is TOP-LEVEL like --verify-standup (the drive spells PETRI_RUN_OPTS before the subcommand)
    ns = prun.build_parser().parse_args(["--serving-path", "lens", "--served-endpoint", str(ep), "print-cells", "--out-root", str(tmp_path)])
    assert str(ns.served_endpoint) == str(ep) and ns.verify_standup is None


# --- a REJECTED request is a hard error, never a shed; an armed pass at zero rows resolves nothing ---------------------

_REJECT_400 = ("Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': "
               "\"'claude-sonnet-5' does not support the `fallbacks` parameter.\"}, 'request_id': 'req_synthetic'}")


def _rejected_sample(sid, *, status_text=_REJECT_400, role="auditor", model="anthropic/claude-sonnet-5", body_words=""):
    """An errored sample the way the SDK leaves a rejected request in the log (the 2026-10-01 incident's shape): ONE errored
    model event carrying the compact status text, the sample error a ModelGenerateError that prepends the whole request body
    (``body_words`` lets a test plant transport-looking words in the scenario's own text) and ends on the exception repr."""
    smp = sample(sid, None, error={"message": "ModelGenerateError('\nRequest:\n{\n  \"messages\": [{\"role\": \"user\", \"content\": \""
                                              + f"Plan the conversation. {body_words}" + "\"}],\n  \"model\": \"claude-sonnet-5\"\n}\n\n"
                                              + f"BadRequestError({status_text!r})')"})
    smp["uuid"] = f"synthetic-{sid}"
    smp["events"] = [{"event": "model", "role": role, "model": model, "timestamp": "2026-10-01T08:31:28.967507+00:00",
                      "error": {"message": status_text}}]
    smp["completed_at"] = "2026-10-01T08:31:31.810327+00:00"
    smp["messages"] = [{"role": "user", "content": "Plan the conversation."}]
    return smp


def _exhausted_sample(sid, *, status_text="Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}"):
    """An errored sample whose ladder genuinely ran out: several errored model events, each the retryable status."""
    smp = sample(sid, None, error={"message": f"ModelGenerateError('\nRequest:\n{{}}\n\nInternalServerError({status_text!r})')"})
    smp["uuid"] = f"synthetic-{sid}"
    smp["events"] = [{"event": "model", "role": "auditor", "model": "anthropic/claude-sonnet-5",
                      "timestamp": f"2026-10-01T08:{31 + i:02d}:00+00:00", "error": {"message": status_text}} for i in range(3)]
    smp["completed_at"] = "2026-10-01T09:12:00+00:00"
    smp["messages"] = [{"role": "assistant", "content": "auditor asks"}, {"role": "tool", "content": "<target_response>t</target_response>"}]
    return smp


@pytest.mark.parametrize("status_text, exhausted, status, etype", [
    (_REJECT_400, False, 400, "invalid_request_error"),
    ("Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}", False, 401, "authentication_error"),
    ("Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-sonnet-5'}}", False, 404, "not_found_error"),
    ("Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'This request would exceed...'}}", True, 429, "rate_limit_error"),
    ("Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}", True, 529, "overloaded_error"),
    ("Error code: 503 service unavailable", True, 503, None),
    ("APIStatusError 500 from the auditor pool", True, 500, None),
    ("APIStatusError 200 overloaded_error", True, None, None),          # Anthropic's status-200 overloaded form: transport evidence
    ("APIConnectionError('Connection error.')", True, None, None),
    ("same body-specific failure", False, None, None),                   # no status, no transport evidence: no ladder claim
])
def test_ladder_exhaustion_reads_the_final_status_and_only_a_retryable_one_or_transport_evidence_is_a_shed(status_text, exhausted, status, etype):
    got = pr.ladder_exhaustion(_rejected_sample("s", status_text=status_text))
    assert (got[0], got[1], got[2]) == (exhausted, status, etype), got
    assert isinstance(got[3], str) and got[3]


def test_the_auditor_chains_all_legs_content_refusal_is_a_shed_but_its_recorded_rejection_is_not():
    """The all-leg refusal (test_petri_auditor_fallback's per-cell report): every configured leg refused this body — no HTTP
    status anywhere, the chain's final_failure content_correlated with every leg used — qualifies. The incident's chain shape —
    final_failure `undetermined`, one leg used of two, the leg's recorded error a 400 — does not, whatever the sample text says."""
    from sisteer.capeval.petri import auditor_fallback as af
    chain = {"version": 1, "configured": [{"model": "anthropic/claude-sonnet-5", "label": "sonnet-5"}, {"model": "openrouter/x-ai/grok-4.6", "label": "grok-4.6"}],
             "used": [{"model": "anthropic/claude-sonnet-5", "label": "sonnet-5", "switch_turn": 1}, {"model": "openrouter/x-ai/grok-4.6", "label": "grok-4.6", "switch_turn": 2}],
             "switches": [], "display": "sonnet-5 → grok-4.6", "final_failure": {"kind": "content_correlated", "all_legs_exhausted": True}}
    smp = _rejected_sample("s", status_text="final refusal"); smp["store"] = {af.AUDITOR_CHAIN_STORE_KEY: chain}
    exhausted, status, _etype, why = pr.ladder_exhaustion(smp)
    assert (exhausted, status) == (True, None) and "every auditor refused" in why
    incident = {**chain, "used": chain["used"][:1], "display": "sonnet-5", "current_index": 0,
                "failures": [{"model": "anthropic/claude-sonnet-5", "error": "ModelGenerateError('Request: {...}\n\nBadRequestError(" + repr(_REJECT_400) + ")')",
                              "discriminator": {"kind": "undetermined"}}],
                "final_failure": {"kind": "undetermined", "all_legs_exhausted": False}}
    smp2 = _rejected_sample("s", status_text="final refusal"); smp2["store"] = {af.AUDITOR_CHAIN_STORE_KEY: incident}
    smp2["events"] = []; smp2["error"] = {"message": "final refusal"}       # only the chain's own record names the status
    assert pr.ladder_exhaustion(smp2)[:3] == (False, 400, "invalid_request_error")
    smp3 = _rejected_sample("s", status_text="final refusal"); smp3["store"] = {af.AUDITOR_CHAIN_STORE_KEY: {**incident, "failures": []}}
    assert pr.ladder_exhaustion(smp3)[0] is False      # an undetermined chain failure with no status is no ladder claim either


def test_a_transport_word_in_the_request_body_is_not_ladder_evidence_for_a_rejected_request():
    """The SDK error prepends the WHOLE request body; a scenario about a server that 'timed out' must not turn its 400 into a shed."""
    smp = _rejected_sample("s", body_words="the deployment timed out and the connection was reset while overloaded")
    exhausted, status, etype, why = pr.ladder_exhaustion(smp)
    assert (exhausted, status, etype) == (False, 400, "invalid_request_error") and "REJECTED" in why
    assert pr._error_tail(pr._sample_error_text(smp)).startswith("BadRequestError(")


def test_the_converter_files_a_rejected_request_as_a_hard_error_never_a_shed_and_the_sidecar_fold_ignores_it(world):
    from test_petri_store_shape import BEHAVIORS, convert
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    rows, entries = convert(world, cell, b, [_rejected_sample(SIDS[0]), sample(SIDS[1], full_dims(b))])
    assert len(rows) == 3 and [e.dimension for e in entries] == list(pr.required_metrics(b))
    for e in entries:
        assert e.shed is None and e.judge_refusal is None and e.hard_error is not None
        assert e.reason.startswith(pr.HARD_ERROR_REASON_PREFIX) and "Error code: 400" in e.reason
        assert (e.hard_error.status, e.hard_error.error_type, e.hard_error.role, e.hard_error.attempts) == (400, "invalid_request_error", "auditor", 1)
        assert e.hard_error.error.startswith("BadRequestError(") and e.hard_error.sample_uuid == f"synthetic-{SIDS[0]}"
    assert pr.auditor_refusals_from_ledger(entries) == []
    rec = pr._ledger_record(entries[0])
    assert "shed" not in rec and "judge_refusal" not in rec and rec["hard_error"]["status"] == 400
    assert pr._ledger_entry_from_record({**rec, "retries": 3}) == entries[0]
    # the exhausted ladder keeps today's path and now records its final status; a record can never claim a rejected status as a shed
    _rows, shed_entries = convert(world, cell, b, [_exhausted_sample(SIDS[0]), sample(SIDS[1], full_dims(b))])
    assert all(e.shed is not None and e.shed.final_status == 529 and e.hard_error is None for e in shed_entries)
    (refusal,) = pr.auditor_refusals_from_ledger(shed_entries)
    assert refusal.shed.final_status == 529 and refusal.shed.kind == pr.SHED_UNDETERMINED
    with pytest.raises(pr.PetriRunnerError, match="not one a ladder retries"):
        pr.AuditorRefusal(behavior=b, scenario_id=SIDS[0], epoch=1, dimensions=tuple(pr.required_metrics(b)),
                          shed=replace(refusal.shed, final_status=400))
    with pytest.raises(pr.PetriRunnerError, match="shed candidate"):
        replace(entries[0].hard_error, status=529)
    # an auditor-refused sidecar written before the field still loads (final_status None)
    old = {k: v for k, v in pr.asdict(refusal.shed).items() if k != "final_status"}
    assert pr._shed_from_record(old).final_status is None


def _rejecting_driver(world, cell, *, status_text=_REJECT_400):
    """Every launched sample comes back REJECTED (the 2026-10-01 pass: one 400 per sample, seconds after launch)."""
    seen = []

    def drive(tasks, *, spec, cell, engine):
        seen.append([(t.behavior_id, tuple(t.scenario_ids) if t.scenario_ids is not None else None) for t in tasks])
        logs = []
        for t in tasks:
            log = make_log(world, cell, t.dir_key, [_rejected_sample(sid, status_text=status_text) for sid in (t.scenario_ids or SIDS)], spec=spec)
            log["location"] = f"/synthetic/{t.behavior_id}.eval"
            logs.append(log)
        return logs
    drive.seen = seen
    return drive


def _shedding_driver(world, cell):
    """Every launched sample's auditor ladder genuinely runs out (retryable statuses) — qualified sheds, zero rows."""
    def drive(tasks, *, spec, cell, engine):
        return [make_log(world, cell, t.dir_key, [_exhausted_sample(sid) for sid in (t.scenario_ids or SIDS)], spec=spec) for t in tasks]
    return drive


def _ledger_sans_retries(store):
    payload = pr.read_failure_ledger(store)
    return [{k: v for k, v in e.items() if k != "retries"} for e in payload["unresolved"]], [e["retries"] for e in payload["unresolved"]], payload


def test_the_armed_pass_whose_every_relaunch_is_rejected_keeps_the_casualties_byte_identical_writes_nothing_terminal_and_the_driver_refuses_by_name(world, tmp_path, monkeypatch):
    """The incident, end to end against the real runner: two overflow casualties armed, every relaunched sample 400s. The casualties
    stand field for field (only the writer's retries counter moves), the ledger lists the rejections, no auditor_refused sidecar and
    no results.json appear, the run summary is INCOMPLETE, and the driver's progress assertion and cmd_run both refuse naming the
    HTTP error — never the replay diagnosis, never COMPLETE."""
    prun = _petri_run()
    spec = _amended(world.spec)
    cell = world.cells["si_safety__psr_kl_all"]
    from test_petri_store_shape import BEHAVIORS
    b0, b1 = BEHAVIORS
    store = _seed_store(world, cell, tmp_path, overflow=[(b0, SIDS[0]), (b1, SIDS[1])])
    before, retries_before, _ = _ledger_sans_retries(store)
    assert len(before) == 2 * len(pr.required_metrics(b0)) and all(e["reason"].startswith(pr.OVERFLOW_REASON_PREFIX) for e in before)
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    armed = prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)
    driver = _rejecting_driver(world, cell)
    result = _run_pass(world, cell, store, tmp_path, driver, spec)
    (tasks,) = driver.seen
    assert sorted(tasks) == sorted([(b0, (SIDS[0],)), (b1, (SIDS[1],))])
    assert result["incomplete"] is True and result["unscored"] == len(before) and result["rows_emitted"] == 0 and result["hard_errors"] == 2
    after, retries_after, payload = _ledger_sans_retries(store)
    assert after == before                                                   # the casualties: byte-identical fields
    assert retries_after == [r + 1 for r in retries_before]                  # the writer counted the (rejected) attempt
    assert [(h["behavior"], h["scenario_id"], h["hard_error"]["status"], h["hard_error"]["error_type"], h["dimensions"]) for h in payload[pr.HARD_ERRORS_KEY]] \
        == sorted([(b0, SIDS[0], 400, "invalid_request_error", sorted(pr.required_metrics(b0))), (b1, SIDS[1], 400, "invalid_request_error", sorted(pr.required_metrics(b1)))])
    assert pr.TERMINAL_WITHHELD_KEY not in payload
    assert not pr.auditor_refused_path(store).is_file() and not store.results_path.exists() and pr.read_auditor_refused(store) == []
    with pytest.raises(prun.DriverError, match="REJECTED by the API") as ei:
        prun._assert_overflow_rerun_progress(store.cell_dir, cell.cell_id, armed, pre_rerolled=0)
    assert "HTTP 400 invalid_request_error" in str(ei.value) and "not a geometry problem" in str(ei.value) and "resolved NONE" not in str(ei.value)
    with pytest.raises(prun.DriverError, match="REJECTED by the API"):
        prun._refuse_rejected_pass(store.cell_dir, cell.cell_id)
    # the next armed pass still finds its casualties (nothing was consumed by the rejection) ...
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    assert len(prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)) == len(before)
    # ... and a pass whose relaunch then SCORES clears the hard-error record from the ledger and completes
    result2 = _run_pass(world, cell, store, tmp_path, _capturing_driver(world, cell), spec)
    assert result2["incomplete"] is False and result2["rows_emitted"] == 2 * len(pr.required_metrics(b0)) and "hard_errors" not in result2
    assert pr.HARD_ERRORS_KEY not in pr.read_failure_ledger(store) and store.results_path.exists()
    prun._refuse_rejected_pass(store.cell_dir, cell.cell_id)   # nothing to refuse


def test_a_plain_resume_that_meets_a_rejected_request_stays_incomplete_with_the_rejection_recorded(world, tmp_path, monkeypatch):
    """Outside the armed pass the same guard holds: a first pass whose one scenario is rejected (no prior record for the key) files
    the hard-error entry itself as unresolved — never terminal, never complete — and cmd_run's refusal reads it."""
    prun = _petri_run()
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    cell = world.cells["si_safety__psr_kl_all"]
    from sisteer.capeval import store as cs
    from test_petri_store_shape import BEHAVIORS
    store = cs.CellStore(tmp_path / "sweep", pr.petri_cell_key(cell))

    def drive(tasks, *, spec, cell, engine):
        logs = []
        for t in tasks:
            samples = [(_rejected_sample(sid) if (t.behavior_id, sid) == (BEHAVIORS[0], SIDS[0]) else sample(sid, full_dims(t.behavior_id))) for sid in SIDS]
            logs.append(make_log(world, cell, t.dir_key, samples, spec=spec))
        return logs
    result = _run_pass(world, cell, store, tmp_path, drive, world.spec)
    assert result["incomplete"] is True and result["unscored"] == len(pr.required_metrics(BEHAVIORS[0])) and result["hard_errors"] == 1
    payload = pr.read_failure_ledger(store)
    assert all(e["reason"].startswith(pr.HARD_ERROR_REASON_PREFIX) and e["hard_error"]["status"] == 400 for e in payload["unresolved"])
    assert not pr.auditor_refused_path(store).is_file() and not store.results_path.exists()
    with pytest.raises(prun.DriverError, match="REJECTED by the API"):
        prun._refuse_rejected_pass(store.cell_dir, cell.cell_id)


def test_the_armed_pass_that_files_zero_rows_and_only_terminal_sheds_withholds_them_and_the_driver_refuses(world, tmp_path, monkeypatch):
    """The zero-resolution floor: every relaunched casualty (a full transcript at the smaller window) comes back a QUALIFIED shed and
    no row is filed — the terminal records are withheld (recorded), the casualties stand, no sidecar, no results.json, and the driver
    refuses as 'resolved NOTHING', not as a replay."""
    prun = _petri_run()
    spec = _amended(world.spec)
    cell = world.cells["si_safety__psr_kl_all"]
    from test_petri_store_shape import BEHAVIORS
    b0 = BEHAVIORS[0]
    store = _seed_store(world, cell, tmp_path, overflow=[(b0, SIDS[0])])
    before, _, _ = _ledger_sans_retries(store)
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    armed = prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)
    result = _run_pass(world, cell, store, tmp_path, _shedding_driver(world, cell), spec)
    assert result["incomplete"] is True and result["rows_emitted"] == 0 and result["terminal_withheld"] == 1 and "hard_errors" not in result
    after, _, payload = _ledger_sans_retries(store)
    assert after == before
    (wa,) = payload[pr.TERMINAL_WITHHELD_KEY]["auditor_refused"]
    assert (wa["behavior"], wa["scenario_id"], wa["shed"]["final_status"], wa["shed"]["kind"]) == (b0, SIDS[0], 529, pr.SHED_UNDETERMINED)
    assert payload[pr.TERMINAL_WITHHELD_KEY]["launched"] == [[b0, SIDS[0]]]
    assert not pr.auditor_refused_path(store).is_file() and not store.results_path.exists()
    with pytest.raises(prun.DriverError, match="resolved NOTHING"):
        prun._assert_overflow_rerun_progress(store.cell_dir, cell.cell_id, armed, pre_rerolled=0)
    # a plain resume (not armed) of a single genuinely shedding scenario keeps today's N-1/N contract: the floor is the armed pass's
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    result2 = _run_pass(world, cell, store, tmp_path, _shedding_driver(world, cell), spec)
    assert result2["incomplete"] is False and result2["rows_emitted"] == 0 and result2["auditor_refused"]["total"] == 1
    assert pr.TERMINAL_WITHHELD_KEY not in pr.read_failure_ledger(store)


def test_a_fallback_judge_outcome_never_lifts_the_floor_for_the_auditor_casualties_and_a_resume_relaunches_them(world, tmp_path, monkeypatch):
    """An armed pass that re-rolls one judge refusal and relaunches TWO overflow casualties; the fallback judge
    refuses the finished transcript again while BOTH casualties' auditors shed, zero rows. The re-roll is recorded as resolved on its
    own entry (stamped refiled_terminal; its record moved to the fallback judge) and lifts nothing for the auditor casualties: their
    sheds are withheld, their records stand byte-identical, no auditor sidecar, no results.json, the pass is INCOMPLETE, the driver
    refuses — and the next armed pass relaunches exactly those two casualties (the re-roll is never re-rolled twice) and completes."""
    prun = _petri_run()
    spec = _amended(world.spec)
    cell = world.cells["si_safety__psr_kl_all"]
    from test_petri_judge_refusal import refused_sample
    from test_petri_store_shape import BEHAVIORS
    b0, b1 = BEHAVIORS
    casualties = [(b0, SIDS[0]), (b1, SIDS[0])]
    store = _seed_store(world, cell, tmp_path, overflow=casualties, judge_refused=[(b1, SIDS[1])])
    before, _, _ = _ledger_sans_retries(store)
    assert len(before) == 2 * len(pr.required_metrics(b0))
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    armed = prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)
    seen = []

    def drive(tasks, *, spec, cell, engine):
        seen.append(sorted((t.behavior_id, sid) for t in tasks for sid in (t.scenario_ids or SIDS)))
        logs = []
        for t in tasks:
            samples = []
            for sid in (t.scenario_ids or SIDS):
                if (t.behavior_id, sid) == (b1, SIDS[1]):
                    smp = refused_sample(sid); smp["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-4-8")]
                    samples.append(smp)
                else:
                    samples.append(_exhausted_sample(sid))
            logs.append(make_log(world, cell, t.dir_key, samples, spec=spec))
        return logs
    result = _run_pass(world, cell, store, tmp_path, drive, spec)
    assert seen == [sorted(casualties + [(b1, SIDS[1])])]
    assert result["rows_emitted"] == 0 and result["incomplete"] is True and result["unscored"] == len(before) and result["terminal_withheld"] == 2
    after, _, payload = _ledger_sans_retries(store)
    assert after == before                                                            # the auditor casualties: intact
    assert sorted((w["behavior"], w["scenario_id"]) for w in payload[pr.TERMINAL_WITHHELD_KEY]["auditor_refused"]) == sorted(casualties)
    assert "1 fallback-judge outcome(s)" in payload[pr.TERMINAL_WITHHELD_KEY]["why"]
    assert not pr.auditor_refused_path(store).is_file() and not store.results_path.exists() and pr.read_auditor_refused(store) == []
    # the judge re-roll IS recorded as resolved on its own entry: stamped, its standing record now the fallback judge's
    (stamp,) = pr.read_judge_rerolled(store)
    assert stamp["outcome"] == "refiled_terminal"
    assert [(r.scenario_id, r.judge_model) for r in pr.read_judge_refused(store)] == [(SIDS[1], "anthropic/claude-opus-4-8")]
    with pytest.raises(prun.DriverError, match="resolved NOTHING"):
        prun._assert_overflow_rerun_progress(store.cell_dir, cell.cell_id, armed, pre_rerolled=0)
    # the casualties are RELAUNCHABLE: the next armed pass's task list is exactly the two of them (the re-roll is spent) ...
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    armed2 = prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)
    assert len(armed2) == len(before)
    driver2 = _capturing_driver(world, cell)
    result2 = _run_pass(world, cell, store, tmp_path, driver2, spec)
    (tasks2,) = driver2.seen
    assert sorted(tasks2) == sorted([(b0, (SIDS[0],)), (b1, (SIDS[0],))])
    # ... and scoring them completes the cell: the withheld record is gone, the ledger settled, results.json written
    assert result2["incomplete"] is False and result2["rows_emitted"] == len(before) and "terminal_withheld" not in result2
    assert pr.TERMINAL_WITHHELD_KEY not in pr.read_failure_ledger(store) and store.results_path.exists()
    prun._assert_overflow_rerun_progress(store.cell_dir, cell.cell_id, armed2, pre_rerolled=1)   # progress: nothing raised


def test_a_pass_whose_only_work_was_a_judge_re_roll_is_outside_the_floor_and_completes(world, tmp_path, monkeypatch):
    """The floor is scoped to AUDITOR casualties (no livelock on the lone re-roll): an armed pass that re-rolls one judge refusal and
    relaunches one overflow casualty, where the casualty SCORES... is the ordinary case; here the casualty is instead left to the judge
    path: zero rows, zero auditor sheds, the fallback judge refuses the re-roll again and the casualty's transcript too — no auditor
    shed anywhere, so nothing is withheld and the cell completes N-2/N on judge-terminal outcomes alone."""
    prun = _petri_run()
    spec = _amended(world.spec)
    cell = world.cells["si_safety__psr_kl_all"]
    from test_petri_judge_refusal import refused_sample
    from test_petri_store_shape import BEHAVIORS
    b0, b1 = BEHAVIORS
    store = _seed_store(world, cell, tmp_path, overflow=[(b0, SIDS[0])], judge_refused=[(b1, SIDS[1])])
    monkeypatch.setenv(pr.PETRI_RERUN_WINDOW_SCALE_ENV, "3"); monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    armed = prun._arm_overflow_rerun(store.cell_dir, cell.cell_id, spec)

    def drive(tasks, *, spec, cell, engine):
        logs = []
        for t in tasks:
            samples = []
            for sid in (t.scenario_ids or SIDS):
                smp = refused_sample(sid); smp["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-4-8")]
                samples.append(smp)
            logs.append(make_log(world, cell, t.dir_key, samples, spec=spec))
        return logs
    result = _run_pass(world, cell, store, tmp_path, drive, spec)
    assert result["rows_emitted"] == 0 and result["incomplete"] is False and "terminal_withheld" not in result
    assert pr.TERMINAL_WITHHELD_KEY not in pr.read_failure_ledger(store) and store.results_path.exists()
    assert sorted((r.behavior, r.scenario_id) for r in pr.read_judge_refused(store)) == sorted([(b0, SIDS[0]), (b1, SIDS[1])])
    assert not pr.zero_resolution_floor(armed=True, rows_emitted=0, new_auditor_terminal=0)
    assert pr.zero_resolution_floor(armed=True, rows_emitted=0, new_auditor_terminal=1)
    assert not pr.zero_resolution_floor(armed=False, rows_emitted=0, new_auditor_terminal=1)
    assert not pr.zero_resolution_floor(armed=True, rows_emitted=3, new_auditor_terminal=1)
    prun._assert_overflow_rerun_progress(store.cell_dir, cell.cell_id, armed, pre_rerolled=0)   # progress: nothing raised


# --- Grok 4.6 as the TERMINAL per-request refusal-fallback JUDGE leg (GROK46 / GROK46_ALIAS above) ------


def _baseline_with(tmp_path, **petri_fields):
    import yaml
    doc = yaml.safe_load(BASELINE_MANIFEST.read_text())
    doc["petri"].update(petri_fields)
    p = tmp_path / "m.yaml"; p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(p)


def test_the_validator_admits_exactly_one_terminal_openrouter_judge_leg(tmp_path):
    """(i): the rendered battery manifest's [opus-4-8, grok-4.6] loads; grok alone (after the default) loads; every other non-Claude
    shape refuses by name -- two non-Claude legs, a non-Claude leg before a Claude one, a non-OpenRouter provider, a non-Claude
    AUDITOR leg -- and the constants the generator mirrors are what the loader uses."""
    assert (cm.PETRI_REFUSAL_FALLBACK_TERMINAL_ROLE, cm.PETRI_REFUSAL_FALLBACK_TERMINAL_PROVIDER) == ("judge", "openrouter/")
    assert cm.load_petri_spec(str(BASELINE_MANIFEST)).judge_refusal_fallback_models == [OPUS48, GROK46]
    assert cm.load_petri_spec(_baseline_with(tmp_path, judge_refusal_fallback_models=[GROK46])).judge_refusal_fallback_models == [GROK46]
    for field, bad, phrase in [
        ("judge_refusal_fallback_models", [GROK46, OPUS48], "must be TERMINAL"),
        ("judge_refusal_fallback_models", [OPUS48, GROK46, "openrouter/x-ai/grok-4.7"], "two non-Claude legs"),
        ("judge_refusal_fallback_models", [OPUS48, "openai/gpt-5"], "off openrouter/"),
        ("judge_refusal_fallback_models", [OPUS48, GROK46_ALIAS], "IMMUTABLE DATED build"),       # Never the mutable alias
        ("auditor_refusal_fallback_models", [SONNET46, GROK46], "first-party Claude ids only"),
    ]:
        with pytest.raises(cm.ManifestError, match=phrase):
            cm.load_petri_spec(_baseline_with(tmp_path, **{field: bad}))


def _answer_output(model, tool="answer"):
    """A structured judge answer: the forced answer tool called (inspect_scout reads the call's arguments as the value)."""
    msg = SimpleNamespace(role="assistant", model=model, tool_calls=[SimpleNamespace(function=tool, arguments={"score": 3})])
    choice = SimpleNamespace(stop_reason="tool_calls", stop_details=None, message=msg)
    return SimpleNamespace(model=model, stop_reason="tool_calls", choices=[choice], metadata=None)


def _judge_wrapper(monkeypatch, calls, key="synthetic-openrouter-credential"):
    """The judge's default wrapped in [opus-4-8, grok-4.6]; the OpenRouter key resolved in-process (stubbed), never from a file."""
    mm = _fake_inspect_model_module(monkeypatch, calls)
    monkeypatch.setattr(af, "resolve_openrouter_api_key", lambda **kw: key)
    primary = mm.FakeModel("anthropic/claude-opus-5", answers=[])
    w = af.make_refusal_fallback_model(primary, "anthropic/claude-opus-5", [OPUS48, GROK46], role="judge")
    opus48, grok = (m for _n, m in w._petri_fallbacks)
    return mm, primary, w, opus48, grok


def _cf(model):
    return _output(model, "content_filter", details=SimpleNamespace(type="refusal", category="bio", model_dump=lambda mode: {"type": "refusal", "category": "bio"}))


def test_the_terminal_grok_leg_scores_a_transcript_every_claude_leg_refused(monkeypatch):
    """(ii)/(iv): opus-5 and opus-4-8 return content_filter; the IDENTICAL request (the same rubric prompt, tools, forced answer
    tool) is re-issued to grok-4.6 over OpenRouter -- resolved with the in-process key, never a file -- and its structured answer is
    returned as the served output, stamped served=grok with both refused attempts; the next request is at opus-5 again."""
    import asyncio
    calls = []
    mm, primary, w, opus48, grok = _judge_wrapper(monkeypatch, calls)
    assert calls == [(OPUS48, {"memoize": False}), (GROK46, {"memoize": False, "api_key": "synthetic-openrouter-credential"})]
    assert w.role == "judge" and grok.role == "judge"
    answer = _answer_output("x-ai/grok-4.6")
    primary.answers = [_cf("claude-opus-5"), _output("claude-opus-5", "tool_calls")]
    opus48.answers = [_cf("claude-opus-4-8")]; grok.answers = [answer]
    args = ("RUBRIC PROMPT", ("answer",), SimpleNamespace(name="answer"), "CFG", None)
    out = asyncio.run(w.generate(*args))
    assert out is answer and grok.seen == [args] and opus48.seen == [args] and out.stop_reason == "tool_calls"
    stamp = out.metadata[af.REFUSAL_FALLBACK_SERVED_KEY]
    assert (stamp["requested"], stamp["served"], stamp["outcome"], stamp["client_side"]) == ("anthropic/claude-opus-5", GROK46, "served", True)
    assert [a["model"] for a in stamp["attempts"]] == ["anthropic/claude-opus-5", OPUS48] and all(af.NON_ANSWER_KEY not in a for a in stamp["attempts"])
    # the served id is attested against the pin -- OpenRouter echoes the ALIAS for a dated request (verified live), verdict alias
    # Grok-fix-3: the alias echo is DECLARED-ONLY -- the request pinned the build, the provider attested nothing; never "attested"
    assert stamp[af.BUILD_ATTESTATION_KEY] == {"requested": GROK46, "pin": "x-ai/grok-4.6-20260810", "family": "x-ai/grok-4.6", "served": "x-ai/grok-4.6",
                                               "verdict": af.BUILD_CLASS_DECLARED, "build_verified": False,
                                               "claim": af.build_claim(GROK46, "x-ai/grok-4.6", af.BUILD_CLASS_DECLARED)}
    out2 = asyncio.run(w.generate(*args))                                   # the next request: the default again, no fallback call
    assert out2 is primary.answers == [] or out2.model == "claude-opus-5"
    assert len(grok.seen) == 1


def test_a_grok_refusal_or_non_answer_is_the_terminal_judge_refusal_filed_under_grok(monkeypatch):
    """(iv): grok's own content_filter -> all_refused, the LAST refusal returned (grok's); grok returning WITHOUT the forced answer
    tool (a textual refusal) or with an OpenAI-style stop_details refusal is marked a returned refusal (content_filter + the reason
    under non_answer) so inspect_scout files the terminal judge refusal; and B4 names the terminal Grok leg AS DECLARED
    (openrouter/..., never anthropic/x-ai/...) from the sample's chronological judge events -- under either spelling of the
    served model."""
    import asyncio
    calls = []
    mm, primary, w, opus48, grok = _judge_wrapper(monkeypatch, calls)
    args = ("RUBRIC PROMPT", ("answer",), SimpleNamespace(name="answer"), "CFG", None)
    # (a) grok refuses outright
    primary.answers = [_cf("claude-opus-5")]; opus48.answers = [_cf("claude-opus-4-8")]; grok.answers = [_cf("x-ai/grok-4.6")]
    out = asyncio.run(w.generate(*args))
    assert out.model == "x-ai/grok-4.6" and out.stop_reason == "content_filter"
    stamp = out.metadata[af.REFUSAL_FALLBACK_SERVED_KEY]
    assert (stamp["outcome"], stamp["served"], [a["model"] for a in stamp["attempts"]]) == ("all_refused", None, ["anthropic/claude-opus-5", OPUS48, GROK46])
    assert af.NON_ANSWER_KEY not in stamp["attempts"][2]
    # (b) grok answers in text, the forced answer tool uncalled -> a refusal of that leg, the reason recorded
    text = _output("x-ai/grok-4.6", "stop")
    primary.answers = [_cf("claude-opus-5")]; opus48.answers = [_cf("claude-opus-4-8")]; grok.answers = [text]
    out = asyncio.run(w.generate(*args))
    assert out is text and out.stop_reason == "content_filter" and out.choices[0].stop_reason == "content_filter"
    stamp = out.metadata[af.REFUSAL_FALLBACK_SERVED_KEY]
    assert stamp["outcome"] == "all_refused" and stamp["attempts"][2][af.NON_ANSWER_KEY] == "forced answer tool 'answer' not called"
    assert out.metadata[af.NON_ANSWER_KEY] == "forced answer tool 'answer' not called"
    # (c) an OpenAI-style refusal (stop_details.type refusal, finish_reason stop) is a refusal too
    oai = _output("x-ai/grok-4.6", "stop", details=SimpleNamespace(type="refusal", category=None, model_dump=lambda mode: {"type": "refusal", "category": None}))
    primary.answers = [_cf("claude-opus-5")]; opus48.answers = [_cf("claude-opus-4-8")]; grok.answers = [oai]
    out = asyncio.run(w.generate(*args))
    assert out.stop_reason == "content_filter" and out.metadata[af.REFUSAL_FALLBACK_SERVED_KEY]["attempts"][2][af.NON_ANSWER_KEY] == "stop_details.type refusal"
    # a CLAUDE leg's textual non-answer is NOT re-marked (the rule is the terminal non-Claude leg's only)
    primary.answers = [_cf("claude-opus-5")]; opus48.answers = [_output("claude-opus-4-8", "stop")]
    out = asyncio.run(w.generate(*args))
    assert out.model == "claude-opus-4-8" and out.stop_reason == "stop" and af.NON_ANSWER_KEY not in (out.metadata or {})
    # the predicate itself
    assert af.non_answer_reason(_answer_output("x"), SimpleNamespace(name="answer")) is None
    assert af.non_answer_reason(_answer_output("x", tool="other"), SimpleNamespace(name="answer")) == "forced answer tool 'answer' not called"
    assert af.non_answer_reason(_output("x", "stop"), "auto") is None and af.non_answer_reason(_cf("x"), SimpleNamespace(name="answer")) is None
    # B4: the refusing judge is the LAST fallback-served judge event, returned as the manifest declares it
    spec = cm.load_petri_spec(str(BASELINE_MANIFEST))
    for served in ("x-ai/grok-4.6", "openrouter/x-ai/grok-4.6"):
        smp = _served_sample("s1")
        smp["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8"),
                         _event("judge", GROK46, served), _event("target", "openai/petri__x", "petri__x")]
        assert pr.refusing_judge_model(smp, spec) == GROK46
    smp = _served_sample("s2")
    smp["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8")]
    assert pr.refusing_judge_model(smp, spec) == OPUS48                   # no grok event: opus-4-8 stays the terminal
    assert pr.sample_role_serving_models({"events": [_event("judge", GROK46, "x-ai/grok-4.6"), _event("judge", GROK46, "openrouter/x-ai/grok-4.6")]})["judge"] == {"x-ai/grok-4.6": 2}
    assert pr.refusal_fallback_allowed(spec)["judge"] == {"claude-opus-5", "claude-opus-4-8", "x-ai/grok-4.6"}


def test_a_grok_hard_error_is_a_fallback_leg_error_never_a_shed(monkeypatch):
    """A 5xx / transport failure on the terminal Grok leg raises RefusalFallbackLegError naming the leg -- the
    converter classifies it a HARD error (the marker rule, whatever the status), never a shed, never a Grok-completed sample."""
    import asyncio
    calls = []
    mm, primary, w, opus48, grok = _judge_wrapper(monkeypatch, calls)
    args = ("RUBRIC PROMPT", ("answer",), SimpleNamespace(name="answer"), "CFG", None)

    class Boom(mm.FakeModel):
        async def generate(self, *a, **k):
            raise RuntimeError("Error code: 503 - {'error': {'message': 'Service Unavailable', 'type': 'server_error'}}")
    boom = Boom(GROK46); w._petri_fallbacks[1] = (GROK46, boom)
    primary.answers = [_cf("claude-opus-5")]; opus48.answers = [_cf("claude-opus-4-8")]
    with pytest.raises(af.RefusalFallbackLegError) as ei:
        asyncio.run(w.generate(*args))
    err = ei.value
    assert (err.leg_model, err.primary_model, err.status, err.error_type) == (GROK46, "anthropic/claude-opus-5", 503, "server_error")
    assert str(err).startswith(af.FALLBACK_LEG_ERROR_MARKER) and [a["model"] for a in err.attempts] == ["anthropic/claude-opus-5", OPUS48]
    smp = _rejected_sample("s", status_text=str(err), role="judge", model=GROK46)
    exhausted, status, etype, why = pr.ladder_exhaustion(smp)
    assert exhausted is False and status == 503 and "fallback leg itself failed" in why and "HTTP 503" in why     # a HARD error, named


def test_bare_model_ids_strip_one_provider_prefix_only():
    assert pr._bare_model_id("anthropic/claude-opus-5") == "claude-opus-5" and pr._bare_model_id("claude-opus-5") == "claude-opus-5"
    assert pr._bare_model_id("openrouter/x-ai/grok-4.6") == pr._bare_model_id("x-ai/grok-4.6") == "x-ai/grok-4.6"
    assert pr._bare_model_id("openai/petri__prompted__safety") == "petri__prompted__safety"


def test_the_judge_seed_documents_the_terminal_grok_leg_without_moving_any_identity():
    """The registry's Petri judge row carries the per-request chain [opus-4-8, grok-4.6] as documentation; fallback_chain is
    not a judge identity field, so judge_id -- and through it every Petri instrument id -- is unchanged, and the battery cells keep their
    recorded chain (eval metadata; provenance verification admits their logs under the amended manifest)."""
    import yaml
    from sisteer.registry import identity
    rows = {j["judge_id"]: j for j in yaml.safe_load((REPO / "configs" / "registry" / "judges.yaml").read_text())["judges"]}
    j = rows["petri_judge_opus5"]
    assert j["fallback_chain"] == ["claude-opus-4-8", GROK46] and "TERMINAL judge" in j["fallback_note"] and "x-ai/grok-4.6-20260810" in j["fallback_note"]
    assert "fallback_chain" not in identity.JUDGE_ID_FIELDS and identity.JUDGE_ID_FIELDS == ("judge_model", "rubric_sha256", "protocol_sha256", "threshold_rule")


# --- the IMMUTABLE dated build is pinned by the request; alias and build are ONE judge; any other build is hard ----

def test_the_served_build_is_attested_against_the_pin_and_any_other_build_is_a_hard_error(monkeypatch):
    """OpenRouter accepts x-ai/grok-4.6-20260810 as a request id (verified: one small request, served by xAI) and echoes the alias
    x-ai/grok-4.6 as the response's model; the wrapper admits the pin itself (exact) or its alias, and a response naming another
    build or model raises RefusalFallbackBuildError -- a RefusalFallbackLegError the converter files as a HARD error (never a score,
    never a refusal of record) -- whether the other build answered or refused."""
    import asyncio
    calls = []
    mm, primary, w, opus48, grok = _judge_wrapper(monkeypatch, calls)
    assert calls[1] == (GROK46, {"memoize": False, "api_key": "synthetic-openrouter-credential"})      # the DATED id is what is requested
    args = ("RUBRIC PROMPT", ("answer",), SimpleNamespace(name="answer"), "CFG", None)
    # exact: the response names the pinned build
    primary.answers = [_cf("claude-opus-5")]; opus48.answers = [_cf("claude-opus-4-8")]; grok.answers = [_answer_output("x-ai/grok-4.6-20260810")]
    out = asyncio.run(w.generate(*args))
    assert out.metadata[af.REFUSAL_FALLBACK_SERVED_KEY][af.BUILD_ATTESTATION_KEY]["verdict"] == "exact"
    # a refusal from the alias build: attested (alias), then the refusal stands as the terminal one
    primary.answers = [_cf("claude-opus-5")]; opus48.answers = [_cf("claude-opus-4-8")]; grok.answers = [_cf("x-ai/grok-4.6")]
    out = asyncio.run(w.generate(*args))
    st = out.metadata[af.REFUSAL_FALLBACK_SERVED_KEY]
    assert st["outcome"] == "all_refused" and st[af.BUILD_ATTESTATION_KEY]["verdict"] == af.BUILD_CLASS_DECLARED
    # mismatch: another model, and another dated build -- answered or refused -- are hard errors naming the leg and the served id
    for served_out in (_answer_output("x-ai/grok-4.7"), _cf("x-ai/grok-4.6-20261105"), _answer_output("openrouter/x-ai/grok-4.5")):
        primary.answers = [_cf("claude-opus-5")]; opus48.answers = [_cf("claude-opus-4-8")]; grok.answers = [served_out]
        with pytest.raises(af.RefusalFallbackBuildError) as ei:
            asyncio.run(w.generate(*args))
        err = ei.value
        assert isinstance(err, af.RefusalFallbackLegError) and err.leg_model == GROK46 and err.served_model == served_out.model
        assert str(err).startswith(af.FALLBACK_LEG_ERROR_MARKER) and "served build mismatch" in str(err) and "x-ai/grok-4.6-20260810" in str(err)
        smp = _rejected_sample("s", status_text=str(err), role="judge", model=GROK46)
        exhausted, status, _etype, why = pr.ladder_exhaustion(smp)
        assert exhausted is False and "fallback leg itself failed" in why                       # a HARD error whatever the (absent) status
    # the predicate and the family rule
    assert af.attest_build(GROK46, _answer_output("x-ai/grok-4.6"))["verdict"] == af.BUILD_CLASS_DECLARED
    assert af.attest_build(GROK46, _answer_output("openrouter/x-ai/grok-4.6-20260810"))["verdict"] == af.BUILD_CLASS_EXACT
    assert af.attest_build(GROK46, SimpleNamespace(model="", choices=[]))["verdict"] == af.BUILD_CLASS_DECLARED   # nothing named: declared-only, not a mismatch
    assert af.model_build_family(GROK46) == af.model_build_family(GROK46_ALIAS) == af.model_build_family("x-ai/grok-4.6-20260810") == "x-ai/grok-4.6"
    assert af.model_build_family("anthropic/claude-opus-4-8") == "claude-opus-4-8" and af.is_dated_build(GROK46) and not af.is_dated_build(GROK46_ALIAS)
    assert pr._bare_model_id(GROK46) == "x-ai/grok-4.6-20260810" and pr._judge_family(GROK46) == "x-ai/grok-4.6"


def test_alias_and_dated_build_are_one_judge_in_provenance_admission_and_refusal_attribution():
    """Under the dated manifest, a sample whose judge events were served as `x-ai/grok-4.6` (the echo), `x-ai/grok-4.6-20260810`
    or `openrouter/x-ai/grok-4.6` is admitted by the provenance walk (an old log stamped [opus-4-8] resumes), and a terminal
    refusal is attributed to the DECLARED dated id under every spelling; a sample served by another build is refused by name."""
    spec = cm.load_petri_spec(str(BASELINE_MANIFEST))
    assert spec.judge_refusal_fallback_models == [OPUS48, GROK46]
    assert pr.refusal_fallback_allowed(spec)["judge"] == {"claude-opus-5", "claude-opus-4-8", "x-ai/grok-4.6"}   # families
    cell = SimpleNamespace(cell_id="synthetic_cell")
    md = {pr.JUDGE_REFUSAL_FALLBACK_METADATA_KEY: [OPUS48], pr.AUDITOR_REFUSAL_FALLBACK_METADATA_KEY: [SONNET46]}   # the OLD stamp
    def smp(sid, served, *, pinned=True, pin=GROK46):
        s_ = _served_sample(sid)
        grok_event = _event_pinned("judge", GROK46, served, pin=pin) if pinned else _event("judge", GROK46, served)
        s_["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8"),
                        grok_event, _event("auditor", "anthropic/claude-sonnet-5", "claude-sonnet-5")]
        return s_
    good = [smp("a", "x-ai/grok-4.6"), smp("b", "x-ai/grok-4.6-20260810"), smp("c", "openrouter/x-ai/grok-4.6")]
    opus_only = _served_sample("o"); opus_only["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8")]
    pr._verify_refusal_fallback_provenance({"samples": good + [opus_only]}, md, spec, cell, "synthetic-dir")   # admitted: pinned requests; the Opus-only sample grandfathered
    for s_ in good:
        assert pr.refusing_judge_model(s_, spec) == GROK46                                            # B4: the DECLARED dated id
    with pytest.raises(pr.PetriRunnerError, match="served by \\['x-ai/grok-4.7'\\], outside the manifest's judge contract"):
        pr._verify_refusal_fallback_provenance({"samples": good + [smp("d", "x-ai/grok-4.7")]}, md, spec, cell, "synthetic-dir")
    # Grok-fix-3: the alias echo WITHOUT a request pinned to the manifest's build is not admitted under it -- an
    # unstamped Grok event, or one pinned to ANOTHER dated build, refuses by name; the Opus-only grandfathering is untouched
    with pytest.raises(pr.PetriRunnerError, match="no request pinned to the manifest's build 'x-ai/grok-4.6-20260810'"):
        pr._verify_refusal_fallback_provenance({"samples": [opus_only, smp("e", "x-ai/grok-4.6", pinned=False)]}, md, spec, cell, "synthetic-dir")
    with pytest.raises(pr.PetriRunnerError, match="no request pinned to the manifest's build"):
        pr._verify_refusal_fallback_provenance({"samples": [smp("f", "x-ai/grok-4.6", pin="openrouter/x-ai/grok-4.6-20250101")]}, md, spec, cell, "synthetic-dir")
    pr._verify_refusal_fallback_provenance({"samples": [opus_only]}, md, spec, cell, "synthetic-dir")   # a first-party-only log needs no stamp
    # the raw served id stays the row's provenance (the tally is by what served, families are for the contract)
    assert pr.sample_role_serving_models(good[0])["judge"] == {"claude-opus-5": 1, "claude-opus-4-8": 1, "x-ai/grok-4.6": 1}
    assert pr.sample_role_serving_models(good[1])["judge"]["x-ai/grok-4.6-20260810"] == 1


def test_one_pin_two_readers_the_manifest_terminal_leg_equals_the_judge_seed_entry():
    """The dated Grok pin is written ONCE per reader and must agree: the battery manifest's terminal judge leg and judges.yaml's last
    fallback_chain entry of petri_judge_opus5 -- and the seed's note records the live fact the wrapper relies on."""
    import yaml
    rows = {j["judge_id"]: j for j in yaml.safe_load((REPO / "configs" / "registry" / "judges.yaml").read_text())["judges"]}
    j = rows["petri_judge_opus5"]
    spec = cm.load_petri_spec(str(BASELINE_MANIFEST))
    assert j["fallback_chain"][-1] == spec.judge_refusal_fallback_models[-1] == GROK46 and af.is_dated_build(j["fallback_chain"][-1])
    assert "echoes the alias" in j["fallback_note"] and "attest_build" in j["fallback_note"]


# --- Grok-fix-3: attestation CLASSES, per-record build identity, the ONE judge-only
# --- re-score mechanism, pinned admission ------------------------------------------------------------------------------------

def _event_pinned(role, requested, served, *, pin=GROK46, verdict=None):
    """A judge event whose output carries the client-side hand-off stamp WITH the build attestation the wrapper writes when a
    non-Claude leg answered or refused (auditor_fallback.attest_build over the served id, against ``pin``)."""
    e = _event(role, requested, served)
    att = af.attest_build(pin, SimpleNamespace(model=served, choices=[SimpleNamespace(message=SimpleNamespace(model=served))]))
    if verdict is not None:
        att["verdict"] = verdict
    e["output"]["metadata"] = {af.REFUSAL_FALLBACK_SERVED_KEY: {"requested": "anthropic/claude-opus-5", "served": pin, "outcome": "served",
                                                                "attempts": [], "client_side": True, af.BUILD_ATTESTATION_KEY: att}}
    return e


def test_grok_fix_3_the_attestation_classes_name_what_the_provider_actually_said():
    """exact: the response names the pinned build (the ONLY verified class); declared_unverified: the request pinned the dated slug and
    the response echoed the alias or named nothing -- declared-only, never an attested build; mismatch: another model or build. The
    record says so in plain words (`claim`) and `build_verified` is True for exact alone."""
    exact = af.attest_build(GROK46, _answer_output("x-ai/grok-4.6-20260810"))
    alias = af.attest_build(GROK46, _answer_output("x-ai/grok-4.6"))
    nothing = af.attest_build(GROK46, SimpleNamespace(model="", choices=[]))
    other = af.attest_build(GROK46, _answer_output("x-ai/grok-4.7"))
    assert (exact["verdict"], alias["verdict"], nothing["verdict"], other["verdict"]) == (af.BUILD_CLASS_EXACT, af.BUILD_CLASS_DECLARED, af.BUILD_CLASS_DECLARED, af.BUILD_CLASS_MISMATCH)
    assert exact["build_verified"] is True and alias["build_verified"] is False and nothing["build_verified"] is False and other["build_verified"] is False
    assert "verified" in exact["claim"] and "not attested by the provider" in alias["claim"] and "echoed the alias" in alias["claim"]
    assert "named no model" in nothing["claim"] and "another build or model" in other["claim"] and "x-ai/grok-4.6-20260810" in alias["claim"]
    assert set(af.BUILD_CLASSES) == {"exact", "declared_unverified", "mismatch"} and "alias" not in af.BUILD_CLASSES


def test_grok_fix_3_rows_and_refusal_records_carry_the_judgments_build_identity_and_first_party_records_stay_byte_identical(world, tmp_path):
    """Every Grok-scored row carries `judge_build` {requested dated id, served id, class}; every Grok-refused record carries the same
    three fields; a first-party-only sample's rows are byte-identical to what the runner wrote before, and a first-party refusal's
    sidecar record has no such keys at all (the reader round-trips both)."""
    from test_petri_judge_refusal import refused_sample
    from test_petri_store_shape import BEHAVIORS
    spec = replace(world.spec, auditor_refusal_fallback_models=[SONNET46], judge_refusal_fallback_models=[OPUS48, GROK46])
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    grok = sample(SIDS[0], full_dims(b))
    grok["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8"), _event_pinned("judge", GROK46, "x-ai/grok-4.6")]
    plain = sample(SIDS[1], full_dims(b)); plain["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5")]
    rows, entries = convert(world, cell, b, [grok, plain], spec=spec)
    assert entries == [] and len(rows) == 6
    epochs = spec.bloom_options.epochs
    g_rows = [r for r in rows if r.item_id == pr._item_id(SIDS[0], 1, epochs)]
    p_rows = [r for r in rows if r.item_id == pr._item_id(SIDS[1], 1, epochs)]
    assert len(g_rows) == len(p_rows) == 3
    for r in g_rows:
        assert json.loads(r.extra_json)[pr.JUDGE_BUILD_KEY] == {"requested": GROK46, "served": "x-ai/grok-4.6", "class": af.BUILD_CLASS_DECLARED}
    for r in p_rows:
        assert pr.JUDGE_BUILD_KEY not in json.loads(r.extra_json)
    # BYTE identity of the first-party rows: the same sample under the pre-Grok chain [opus-4-8]
    # serializes to the identical bytes the store writes (store.append_row: json.dumps(asdict(row), sort_keys=True, allow_nan=False))
    import dataclasses as _dc
    from sisteer.capeval import store as _cs
    rows_pre, _ = convert(world, cell, b, [plain], spec=replace(spec, judge_refusal_fallback_models=[OPUS48]))
    # the bytes READ BACK from rows.jsonl after CellStore.append_row (the write stamp pinned so the two writes are comparable)
    def written_lines(rows_, where):
        st = _cs.CellStore(tmp_path / where, cell.petri_run_key)
        for r in rows_:
            st.append_row(_dc.replace(r, created_at=0.0))
        return st.rows_path.read_bytes().splitlines()
    assert written_lines(rows_pre, "pre") == written_lines(p_rows, "post") and all(b"judge_build" not in line for line in written_lines(p_rows, "post2"))
    assert pr.judge_build_attestations(grok) == [{"requested": GROK46, "served": "x-ai/grok-4.6", "class": af.BUILD_CLASS_DECLARED}] and pr.judge_build_attestations(plain) == []
    # refusal records
    gref = refused_sample(SIDS[0]); gref["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8"), _event_pinned("judge", GROK46, "x-ai/grok-4.6")]
    oref = refused_sample(SIDS[1]); oref["events"] = [_event("judge", "anthropic/claude-opus-5", "claude-opus-5"), _event("judge", OPUS48, "claude-opus-4-8")]
    _rows, e2 = convert(world, cell, b, [gref, oref], spec=spec)
    recs = {r.scenario_id: r for r in pr.judge_refusals_from_ledger(e2)}
    assert recs[SIDS[0]].judge_model == GROK46 and (recs[SIDS[0]].judge_requested, recs[SIDS[0]].judge_served, recs[SIDS[0]].judge_build_class) == (GROK46, "x-ai/grok-4.6", af.BUILD_CLASS_DECLARED)
    assert recs[SIDS[1]].judge_model == OPUS48 and (recs[SIDS[1]].judge_requested, recs[SIDS[1]].judge_served, recs[SIDS[1]].judge_build_class) == (None, None, None)
    store = SimpleNamespace(cell_dir=tmp_path / "cell")
    payload = pr.write_judge_refused(store, cell, list(recs.values()))
    by_sid = {x["scenario_id"]: x for x in payload[pr.JUDGE_REFUSED_OUTCOME]}
    assert not (set(pr.JUDGE_BUILD_FIELDS) & set(by_sid[SIDS[1]]))                                  # first-party record: byte-identical shape
    # BYTE identity of the first-party sidecar record: the record as the previous writer serialized it
    # (asdict | dimensions-as-list | schema, through _write_json_atomic's json.dumps(indent=1, sort_keys=True, default=str))
    import dataclasses as _dc
    from sisteer.capeval import store as _cs
    # the FILE bytes -- the sidecar write_judge_refused produces for the first-party record alone, against the previous writer's
    # payload (its record = asdict | dimensions-as-list | schema, no build fields) written through the same _write_json_atomic
    import time as _time
    monkeypatch_time = 1700000000.0
    old_time = pr.time.time
    pr.time.time = lambda: monkeypatch_time
    try:
        fp_store = SimpleNamespace(cell_dir=tmp_path / "fp")
        payload_new = pr.write_judge_refused(fp_store, cell, [recs[SIDS[1]]])
    finally:
        pr.time.time = old_time
    previous_record = ({k: v for k, v in _dc.asdict(recs[SIDS[1]]).items() if k not in pr.JUDGE_BUILD_FIELDS}
                       | {"dimensions": list(recs[SIDS[1]].dimensions), "schema": pr.JUDGE_REFUSED_SCHEMA})
    previous_payload = dict(payload_new) | {pr.JUDGE_REFUSED_OUTCOME: [previous_record]}
    _cs._write_json_atomic(tmp_path / "fp_prev" / "petri_judge_refused.json", previous_payload)
    assert pr.judge_refused_path(fp_store).read_bytes() == (tmp_path / "fp_prev" / "petri_judge_refused.json").read_bytes()
    assert set(by_sid[SIDS[1]]) == {"behavior", "scenario_id", "epoch", "dimensions", "reason", "judge_model", "sample_id", "sample_uuid", "eval_log", "auditor_chain", "outcome", "schema"}
    assert (by_sid[SIDS[0]]["judge_requested"], by_sid[SIDS[0]]["judge_served"], by_sid[SIDS[0]]["judge_build_class"]) == (GROK46, "x-ai/grok-4.6", af.BUILD_CLASS_DECLARED)
    back = {r.scenario_id: r for r in pr.read_judge_refused(store)}
    assert back[SIDS[0]].judge_build_class == af.BUILD_CLASS_DECLARED and back[SIDS[1]].judge_build_class is None and back[SIDS[1]] == recs[SIDS[1]]
    # the tally the footing carries: by class / requested / served
    assert pr.judge_build_tally([json.loads(r.extra_json)[pr.JUDGE_BUILD_KEY] for r in g_rows[:1]]) == {"by_class": {af.BUILD_CLASS_DECLARED: 1}, "requested": {GROK46: 1}, "served": {"x-ai/grok-4.6": 1}}


def test_grok_fix_3_the_judge_only_rescore_selects_exactly_the_refusals_the_terminal_leg_never_judged(world):
    """The baseline shape of 2026-10-01: four cyber refusals filed under opus-5 (the pre-B3 converter's label) whose samples opus-5 AND
    opus-4-8 refused. Under the dated-Grok chain the selector picks EXACTLY those four -- by the sample's judge provenance, never the
    filed label -- and leaves standing a refusal the Grok leg itself issued, an item the ledger names (re-rolled or re-scored), and
    every record under a chain whose terminal leg already refused it; a record without provenance is never selected. Auditor refusals
    and hard errors are not JudgeRefusal records and cannot be candidates. The overflow pass's selector reads the same ledger."""
    spec = replace(world.spec, auditor_refusal_fallback_models=[SONNET46], judge_refusal_fallback_models=[OPUS48, GROK46])
    cyber = ("cyber_a", "cyber_b", "cyber_c", "cyber_d")
    four = [_jr("self_preservation", s) for s in cyber]
    grok_refused = _jr("self_preservation", "already_grok", judge=GROK46)
    rerolled_once = _jr("delusion_sycophancy", "rerolled_once")
    rescored_once = _jr("delusion_sycophancy", "rescored_once")
    prov = {("self_preservation", s): ["claude-opus-5", "claude-opus-4-8"] for s in cyber}
    prov[("self_preservation", "already_grok")] = ["claude-opus-5", "claude-opus-4-8", "x-ai/grok-4.6"]      # the alias echo IS the Grok family
    prov[("delusion_sycophancy", "rerolled_once")] = ["claude-opus-5"]
    prov[("delusion_sycophancy", "rescored_once")] = ["claude-opus-5", "claude-opus-4-8"]
    by = lambda r: prov[(r.behavior, r.scenario_id)]   # noqa: E731
    ledger = [{"behavior": "delusion_sycophancy", "scenario_id": "rerolled_once", "epoch": 1, "rerolled_by": "overflow-rerun pass"},
              {"behavior": "delusion_sycophancy", "scenario_id": "rescored_once", "epoch": 1, "rescored_by": "petri_rescore"}]
    standing, chosen = pr.rescorable_judge_refusals(four + [grok_refused, rerolled_once, rescored_once], spec, by, ledger)
    assert [r.scenario_id for r in chosen] == list(cyber)
    assert [r.scenario_id for r in standing] == ["already_grok", "rerolled_once", "rescored_once"]
    four48 = [_jr("self_preservation", s, judge=OPUS48) for s in cyber]                             # the post-B3 label: the same four
    assert [r.scenario_id for r in pr.rescorable_judge_refusals(four48, spec, by, [])[1]] == list(cyber)
    assert pr.rescorable_judge_refusals(four, world.spec, by, []) == (four, [])                      # no fallback declared: nothing moves
    assert pr.rescorable_judge_refusals(four, replace(spec, judge_refusal_fallback_models=[OPUS48]), by, []) == (four, [])   # opus-4-8 already refused them
    assert pr.rescorable_judge_refusals(four[:1], spec, lambda r: [], [])[1] == []                   # no provenance: never selected
    assert pr.rerollable_judge_refusals([rescored_once], spec, ledger) == ([rescored_once], [])     # one ledger for both mechanisms
    assert pr.rerollable_judge_refusals([rescored_once], spec, []) == ([], [rescored_once])


def test_grok_fix_3_the_sidecar_keeps_the_rescored_list_across_rewrites_like_the_rerolled_one(world, tmp_path):
    cell = world.cells["si_safety__psr_kl_all"]
    store = SimpleNamespace(cell_dir=tmp_path / "cell")
    rec = _jr("self_preservation", SIDS[1], judge=GROK46)
    entries = [{"behavior": "self_preservation", "scenario_id": SIDS[0], "epoch": 1, "rescored_by": "petri_rescore", "revision": "ab" * 8}]
    payload = pr.write_judge_refused(store, cell, [rec], rescored=entries)
    assert payload[pr.JUDGE_RESCORED_KEY] == entries and pr.read_judge_rescored(store) == entries and pr.read_judge_rerolled(store) == []
    pr.write_judge_refused(store, cell, [rec])                                                       # a rewrite without the argument keeps it
    assert pr.read_judge_rescored(store) == entries
    assert pr.write_judge_refused(store, cell, [rec], rescored=[]).get(pr.JUDGE_RESCORED_KEY) is None
