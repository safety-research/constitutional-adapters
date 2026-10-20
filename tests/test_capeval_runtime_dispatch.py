"""Runtime dispatch fixes from the r1 code review (tier1-runtime findings
#2/#7/#8 + simplify-pass #5/#7), base lane only:

* the retired in-process vLLM engine: a manifest that names it is refused
  by name when the engine is built (it ran only the removed tier-1 task list).
* run_target pairing (#7): legs join by item_id — duplicates and set
  mismatches refuse, never a positional zip.
* run_target order (#8): the prompt ORDER is a deterministic function of
  the seed (attempt 1's fresh seed = a genuinely different order), while
  the selected item SET stays the canonical first n.
* artifact paths (simplify #5): runtime uses the manifest's artifact path
  VERBATIM — no cwd-relative re-resolution.
* run_tier1 resume (#3/#4): resume keys on a validated samples.jsonl.
* run_tier1 token kinds: a prompt-transforming arm is refused before any
  generation, because the HumanEval+ prompts are rendered untransformed.
"""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval import runtime as rt
from sisteer.capeval import store as cs

FIXTURES = Path(__file__).parent / "fixtures" / "capeval"
REV = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch):
    monkeypatch.setattr(rt, "_chat_template", lambda m: "TEST-TEMPLATE")
    monkeypatch.delenv("CAPEVAL_CALIBRATION_JSON", raising=False)


def _vllm_manifest():
    return cm.Manifest(
        sweep="vllm_disp", mode="calibration", base="test-org/tiny",
        base_revision=REV, engine="vllm", tiers=[0, 1], confirmatory=False,
        target=None, margins={},
        ablations=[cm.AblationSpec(name="none", kind="none",
                                   params={"trials": 2})])


def test_make_engine_refuses_the_retired_vllm_engine_by_name():
    """The in-process vLLM engine loaded through the removed tier-1 harness's
    wrapper; a manifest that still names it refuses before any model load,
    naming the engine and the served path that replaces it."""
    with pytest.raises(rt.RuntimeError_, match="engine 'vllm' is retired") as exc:
        rt.make_engine(_vllm_manifest())
    assert "served_capability_run.py" in str(exc.value)
    assert rt.RETIRED_ENGINE_VLLM == "vllm"
    assert "tier1.yaml" not in rt.HASHED_CONFIG_FILES


def _cell(index=0, arm="none", kind="none", coeff=None, trial=0,
          tiers=(0, 1)):
    return SimpleNamespace(index=index, arm=arm, kind=kind, coeff=coeff,
                           trial=trial, tiers=tuple(tiers),
                           time_class="full")


def _store(tmp_path, engine="vllm", arm="none", kind="none", coeff=None):
    row_constants = {
        "sweep": "vllm_disp", "run_id": "r", "model_id": "test-org/tiny",
        "model_revision": REV, "arm_name": arm, "kind": kind, "coeff": coeff,
        "intervention_hash": "a1b2c3d4e5f60718", "engine": engine,
        "trial_id": 0, "seed": 0, "attempt": 0,
        "harness_config_hash": "0f1e2d3c4b5a6978",
        "env_digest": "99aabbccddeeff00", "slurm_job_id": None,
    }
    return cs.CellStore(tmp_path, f"{arm}__tier01__t0__a0__a1b2c3d4__0f1e2d3c",
                        row_constants=row_constants)


def test_run_tier1_hf_resume_validates_samples(tmp_path, monkeypatch):
    """Review tier1 #3/#4: resume keys on samples.jsonl (atomic ⇒
    existence means complete) and re-validates the pinned 164-ID set — a
    truncated file refuses instead of quietly skipping generation."""
    from sisteer.capeval.tiers import evalplus_runner, parity

    monkeypatch.setattr(parity, "run", lambda *a, **k: {})
    monkeypatch.setattr(rt, "_arm_handle", lambda e, m, c: None)

    class HfEngine:
        tokenizer = object()

    m = cm.Manifest(
        sweep="hf_resume", mode="calibration", base="test-org/tiny",
        base_revision=REV, engine="hf", tiers=[0, 1], confirmatory=False,
        target=None, margins={},
        ablations=[cm.AblationSpec(name="none", kind="none")])
    store = _store(tmp_path, engine="hf")
    # complete samples.jsonl -> generation skipped, still awaiting the wave
    evalplus_runner.write_samples_jsonl(
        Path(store.cell_dir) / "samples.jsonl",
        [{"task_id": f"HumanEval/{i}", "solution": "x"} for i in range(164)])
    summary = rt.run_tier1(HfEngine(), m, _cell(), store, seed=0)
    assert summary["evalplus_generation"] == "already complete (resume)"
    assert summary["awaiting_execution"] is True
    # a TRUNCATED samples.jsonl refuses loudly (tier1 #4)
    evalplus_runner.write_samples_jsonl(
        Path(store.cell_dir) / "samples.jsonl",
        [{"task_id": f"HumanEval/{i}", "solution": "x"} for i in range(163)])
    with pytest.raises(evalplus_runner.EvalPlusSeamError, match="164"):
        rt.run_tier1(HfEngine(), m, _cell(), store, seed=0)


class _TokArm:
    def __init__(self, kind, transform=None, name="arm-t"):
        self.kind, self.name, self.intervention_hash = kind, name, "ab" * 8
        self._transform = transform or (lambda s: s)

    def transform_prompt(self, s):
        return self._transform(s)


class _StubArm(_TokArm):
    def transform_prompt(self, s):
        raise NotImplementedError


def test_run_tier1_refuses_a_token_kind_arm_before_any_generation(tmp_path, monkeypatch):
    """The HumanEval+ generation renders evalplus's own prompts and never applies
    the arm's transform_prompt, so an arm whose intervention lives in the prompt
    is refused by name before anything generates: by kind name, or by a
    transform that is not the identity. Every other arm passes."""
    from sisteer.capeval.tiers import evalplus_runner, parity

    def no_generation(**kw):
        raise AssertionError("generation ran for a refused arm")

    monkeypatch.setattr(evalplus_runner, "generate_samples", no_generation)
    monkeypatch.setattr(parity, "run", lambda *a, **k: {})

    class HfEngine:
        tokenizer = object()

    m = cm.Manifest(
        sweep="tok_refuse", mode="calibration", base="test-org/tiny",
        base_revision=REV, engine="hf", tiers=[0, 1], confirmatory=False,
        target=None, margins={},
        ablations=[cm.AblationSpec(name="none", kind="none")])
    store = _store(tmp_path, engine="hf")
    refused = ((_TokArm("token"), "token kind"), (_TokArm("token_bos"), "token kind"),
               (_TokArm("token_faithful"), "token kind"), (_StubArm("token"), "token kind"),
               (_TokArm("steering", transform=lambda s: "<t> " + s), "transforms prompts"))
    for arm, match in refused:
        monkeypatch.setattr(rt, "_arm_handle", lambda e, mm, c, a=arm: a)
        with pytest.raises(rt.TokenKindRefusal, match=match):
            rt.run_tier1(HfEngine(), m, _cell(), store, seed=0)
    assert issubclass(rt.TokenKindRefusal, rt.RuntimeError_)
    for arm in (None, _TokArm("steering"), _StubArm("steering")):
        rt.refuse_token_kinds(arm)


# ------------------------------------------------------------- run_target


class _Leg:
    def __init__(self, item_ids, values, metric="si_compliance"):
        self.item_ids = tuple(item_ids)
        self.values = tuple(values)
        self.metric = metric
        self.params_resolved = {}


def _sweep_manifest():
    return cm.Manifest(
        sweep="tgt", mode="sweep", base="test-org/tiny",
        base_revision=REV, engine="hf", tiers=[0, 1], confirmatory=False,
        margins={},
        target=cm.TargetSpec(
            primaries=[cm.PrimarySpec(metric="si_compliance",
                                      plugin="sisteer_compliance",
                                      params={"si": "number_words",
                                              "n_prompts": 4},
                                      direction="up", min_effect=0.1)],
            secondary=[], off_target_exclusions=[],
            relevant_corpus=cm.CorpusRef(
                slice={"first": 2, "field": "prompt"},
                path=str(FIXTURES / "pairs_synthetic.jsonl"))),
        ablations=[cm.AblationSpec(
            name="arm-a", kind="steering",
            artifact=str(FIXTURES / "vector_synthetic.safetensors"),
            layer=15, coeffs=[2.0])])


def _target_world(monkeypatch, plugin):
    from sisteer.capeval import corpora, targets

    prompts = corpora.CorpusSlice(
        name="alpaca_holdout", kind="prompts",
        item_ids=tuple(f"p{i}" for i in range(6)),
        items=tuple(f"prompt {i}" for i in range(6)), spec={},
        slice_hash="cd" * 8)
    monkeypatch.setattr(corpora, "load_pins", lambda: {})
    monkeypatch.setattr(corpora, "alpaca_holdout_prompts",
                        lambda pins: prompts)
    monkeypatch.setattr(rt, "_arm_handle", lambda e, m, c: object())
    monkeypatch.setattr(
        targets, "resolve_target",
        lambda m: [(m.target.primaries[0], plugin)])


def test_run_target_pairs_by_item_id_not_position(tmp_path, monkeypatch):
    """Review tier1 #7: legs returning the same items in different orders
    pair correctly by id; a missing item on one leg refuses."""
    calls = []

    def plugin(engine, arm, primary, prompts):
        calls.append(list(prompts.item_ids))
        if arm is not None:  # arm leg: shuffled order, values keyed to id
            ids = list(reversed(prompts.item_ids))
            return _Leg(ids, [1.0 if i == "p0" else 0.0 for i in ids])
        ids = list(prompts.item_ids)
        return _Leg(ids, [0.0 for _ in ids])

    _target_world(monkeypatch, plugin)
    m = _sweep_manifest()
    store = _store(tmp_path, engine="hf", arm="arm-a", kind="steering", coeff=2.0)
    summary = rt.run_target(object(), m, _cell(arm="arm-a", kind="steering",
                                                coeff=2.0), store, seed=100)
    p = summary["primaries"]["si_compliance"]
    # delta = mean over id-paired diffs: only p0 moved -> 1/4
    assert p["delta"] == pytest.approx(0.25)
    assert p["n"] == 4  # params.n_prompts, not the full slice


def test_run_target_refuses_mismatched_leg_items(tmp_path, monkeypatch):
    def plugin(engine, arm, primary, prompts):
        ids = list(prompts.item_ids)
        if arm is not None:
            ids = ids[:-1] + ["pX"]  # one foreign item
        return _Leg(ids, [0.0] * len(ids))

    _target_world(monkeypatch, plugin)
    m = _sweep_manifest()
    store = _store(tmp_path, engine="hf", arm="arm-a", kind="steering", coeff=2.0)
    with pytest.raises(rt.RuntimeError_, match="item sets differ"):
        rt.run_target(object(), m, _cell(arm="arm-a", kind="steering",
                                          coeff=2.0), store, seed=100)


def test_run_target_order_derives_from_seed(tmp_path, monkeypatch):
    """Review tier1 #8: attempt 1's fresh seed must rerun the target
    under a DIFFERENT prompt order (same item set)."""
    seen_orders = {}

    def plugin(engine, arm, primary, prompts):
        seen_orders.setdefault(len(seen_orders), list(prompts.item_ids))
        return _Leg(list(prompts.item_ids), [0.0] * len(prompts.item_ids))

    _target_world(monkeypatch, plugin)
    m = _sweep_manifest()
    store = _store(tmp_path, engine="hf", arm="arm-a", kind="steering", coeff=2.0)
    rt.run_target(object(), m, _cell(arm="arm-a", kind="steering", coeff=2.0),
                  store, seed=100)
    order_a0 = seen_orders[0]
    seen_orders.clear()
    store2 = cs.CellStore(tmp_path / "a1", store.key,
                          row_constants=store.row_constants)
    rt.run_target(object(), m, _cell(arm="arm-a", kind="steering", coeff=2.0),
                  store2, seed=100 + 10000)
    order_a1 = seen_orders[0]
    assert sorted(order_a0) == sorted(order_a1)  # same canonical item SET
    assert order_a0 != order_a1                  # different seed -> new order
    # and the same seed reproduces its order exactly (deterministic)
    seen_orders.clear()
    store3 = cs.CellStore(tmp_path / "re", store.key,
                          row_constants=store.row_constants)
    rt.run_target(object(), m, _cell(arm="arm-a", kind="steering", coeff=2.0),
                  store3, seed=100)
    assert seen_orders[0] == order_a0


def test_artifact_paths_used_verbatim(tmp_path):
    """Simplify-pass #5: the loader resolves artifact paths at load; the
    runtime must not re-resolve relative to whatever cwd it has."""
    ab = cm.AblationSpec(name="a", kind="steering",
                         artifact="definitely/not/absolute.safetensors",
                         layer=15, coeffs=[2.0])
    with pytest.raises(rt.RuntimeError_, match="load_manifest"):
        rt._artifact_paths(ab)
    good = cm.AblationSpec(
        name="a", kind="steering",
        artifact=str(FIXTURES / "vector_synthetic.safetensors"),
        layer=15, coeffs=[2.0])
    p, sc = rt._artifact_paths(good)
    assert p.is_absolute() and p.is_file() and sc.suffix == ".json"
