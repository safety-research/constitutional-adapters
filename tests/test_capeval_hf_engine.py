"""HF-engine hygiene tests (arm A → arm B → base ≡
base alone with logits bit-equal; zero-coeff ≡ base; sham ≡ base;
per-call ctx entry counted; weight-noise apply→restore
exact; kind/layer validation refusals). The engine-mismatch PAIRING
refusal lives in the fingerprint — this engine's obligation,
asserted here and in the tier0 e2e, is that its name/rows carry
engine="hf" correctly.

Base lane only; model = the conftest tiny_llama (CPU, float32)."""

import json

import pytest
import torch

from sisteer.capeval.manifest import AblationSpec, ManifestError
from sisteer.capeval.serving.hf_engine import EngineError, HFEngine
from sisteer.capeval.tiers import tier0 as t0
from sisteer.vectors import SteeringVector

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # conftest pin


@pytest.fixture(scope="module")
def engine(tiny_llama):
    model, tok = tiny_llama
    eng = HFEngine(TINY_LLAMA, revision=None, preloaded=(model, tok))
    eng.load()
    return eng


@pytest.fixture(scope="module")
def vector_artifact(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    sv = SteeringVector(
        vectors={1: torch.randn(16, generator=torch.Generator().manual_seed(3))
                 * 2.0},
        method="distill_soft", model_name=TINY_LLAMA, instruction_id="test")
    path = d / "vec.safetensors"
    sv.save(path)
    return path


@pytest.fixture(scope="module")
def corpus(engine):
    texts = ["The river runs east from the high meadow past the mill.",
             "To bake a loaf you need flour, water, salt, and yeast.",
             "The observatory sits above the fog line on the ridge."]
    return t0.windows_from_texts(engine.tokenizer, "hyg", texts,
                                 window_len=12, min_window_tokens=4)


def _fixed_input(engine):
    tok = engine.tokenizer
    return tok("The quick brown fox jumps over the lazy dog",
               return_tensors="pt")["input_ids"]


def _logits(engine, ids):
    with torch.no_grad():
        return engine.model(input_ids=ids).logits.clone()


def _ctrl(name, kind, seed=7, **params):
    return AblationSpec(name=name, kind=kind, seed=seed, params=params,
                        coeffs=[1.0])


# ---------------------------------------------------------------------------
# hook-state hygiene: arm A -> arm B -> base == base alone (bit-equal)
# ---------------------------------------------------------------------------


def test_arm_a_then_arm_b_then_base_is_bit_equal(engine, vector_artifact,
                                                 corpus):
    ids = _fixed_input(engine)
    baseline = _logits(engine, ids)

    arm_a = engine.arm(AblationSpec(name="a", kind="steering",
                                    artifact=str(vector_artifact), layer=1,
                                    coeffs=[2.0]), 2.0)
    arm_b = engine.arm(_ctrl("b", "control:random_vector", layer=1,
                             norm_target=4.0), 1.0)
    # run real scored passes under both arms (hooks registered/removed
    # per item forward), then confirm the base model is untouched
    a_scores = [p.token_nll for p in engine.score_corpus(corpus, arm_a)]
    b_scores = [p.token_nll for p in engine.score_corpus(corpus, arm_b)]
    base_scores = [p.token_nll for p in engine.score_corpus(corpus, None)]
    assert a_scores != base_scores  # the arms actually did something
    assert b_scores != base_scores
    assert torch.equal(_logits(engine, ids), baseline)


def test_zero_coeff_is_bit_equal_to_base(engine, vector_artifact):
    ids = _fixed_input(engine)
    baseline = _logits(engine, ids)
    arm = engine.arm(_ctrl("z", "control:zero_coeff",
                           artifact=str(vector_artifact)), 0.0)
    with arm.ctx():
        steered = _logits(engine, ids)
    assert torch.equal(steered, baseline)
    assert torch.equal(_logits(engine, ids), baseline)  # and clean after


def test_sham_hook_is_bit_equal_to_base(engine):
    ids = _fixed_input(engine)
    baseline = _logits(engine, ids)
    arm = engine.arm(_ctrl("s", "control:sham_hook", layer=1), 1.0)
    with arm.ctx():
        shammed = _logits(engine, ids)
    assert torch.equal(shammed, baseline)
    assert torch.equal(_logits(engine, ids), baseline)


# ---------------------------------------------------------------------------
# ctx() is entered FRESH per call, once per forward/generate
# ---------------------------------------------------------------------------


def test_ctx_entered_fresh_per_call(engine, vector_artifact, corpus):
    arm = engine.arm(AblationSpec(name="a", kind="steering",
                                  artifact=str(vector_artifact), layer=1,
                                  coeffs=[2.0]), 2.0)
    assert arm.ctx_entries == 0
    list(engine.score_corpus(corpus, arm))
    assert arm.ctx_entries == len(corpus.items)  # one ctx per item forward
    engine.generate(["Hello there.", "Count to three."], arm,
                    {"max_new_tokens": 4})
    assert arm.ctx_entries == len(corpus.items) + 1  # ONE per generate call
    # and each entry is a NEW context-manager object (never a reuse)
    c1, c2 = arm.ctx(), arm.ctx()
    assert c1 is not c2


# ---------------------------------------------------------------------------
# weight-mutating controls: exact restore from pinned CPU copies
# ---------------------------------------------------------------------------


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


@pytest.mark.parametrize("kind,params,coeff", [
    ("control:weight_noise", {}, 3e-3),
    ("control:noise_lora", {"rank": 4}, 1e-3),
])
def test_weight_mutation_apply_restore_exact(engine, kind, params, coeff):
    model = engine.model
    before = _snapshot(model)
    ids = _fixed_input(engine)
    baseline = _logits(engine, ids)
    arm = engine.arm(_ctrl("w", kind, **params), coeff)
    with arm.ctx():
        during = _snapshot(model)
        changed = [n for n in before if not torch.equal(before[n], during[n])]
        assert changed, "the damage arm changed nothing"
    after = _snapshot(model)
    for n in before:
        assert torch.equal(before[n], after[n]), f"{n} not restored exactly"
    assert torch.equal(_logits(engine, ids), baseline)


def test_weight_mutation_restores_on_exception(engine):
    model = engine.model
    before = _snapshot(model)
    arm = engine.arm(_ctrl("w", "control:weight_noise"), 3e-3)
    with pytest.raises(RuntimeError, match="boom"):
        with arm.ctx():
            raise RuntimeError("boom")
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p.detach()), f"{n} leaked after crash"


# ---------------------------------------------------------------------------
# atom selection: deterministic top-k + masked-logsumexp tail (§4.3)
# ---------------------------------------------------------------------------


def test_atom_selection_matches_independent_recomputation(engine, corpus):
    item = corpus.items[0]
    got = list(engine.score_corpus(
        t0.WindowedCorpus("hyg", (item,)), None, 3))
    with torch.no_grad():
        logits = engine.model(
            input_ids=torch.tensor([list(item.input_ids)])).logits[0]
    for ps, pos in zip(got, item.score_positions):
        lp = torch.log_softmax(logits[pos - 1].double(), dim=-1)
        # stable descending sort == top-k by prob, ties by ascending id
        vals, order = torch.sort(lp, descending=True, stable=True)
        assert torch.equal(ps.atom_probs.ids.long(), order[:3])
        assert torch.equal(ps.atom_probs.logp, vals[:3])
        masked = lp.scatter(0, order[:3], float("-inf"))
        assert ps.atom_probs.logp_tail == pytest.approx(
            float(torch.logsumexp(masked, dim=-1)), abs=0)
        assert ps.atom_probs.ids.dtype == torch.int32
        assert ps.atom_probs.logp.dtype == torch.float64


def test_score_corpus_refuses_misaligned_atom_ids(engine, corpus):
    with pytest.raises(EngineError, match="mismatch"):
        list(engine.score_corpus(corpus, None,
                                 torch.zeros(2, 3, dtype=torch.int32)))


def test_score_corpus_refuses_non_all_position_policy(engine, corpus,
                                                      vector_artifact):
    """BLOCKER: teacher-forced scoring forwards the full sequence, so a
    selector policy ('last') steers the final input token — not the
    predictor position — and can be exactly base-equivalent.
    Every policy but 'all' is refused until the v1 selector-aware
    scorer exists. (This used to build a positions="response" vector arm
    — the pre-2026-08-28 override that lacked the boundary wiring and
    steered EVERY position, the prompt_len-0 trap; that spelling now
    refuses at ARM BUILD on a headerless tokenizer, and the response
    mask has its own tests in test_response_mask_capabilities.py.)"""
    arm = engine.arm(AblationSpec(name="a", kind="steering",
                                  artifact=str(vector_artifact), layer=1,
                                  coeffs=[2.0], positions="last"), 2.0)
    with pytest.raises(EngineError, match="v1"):
        list(engine.score_corpus(corpus, arm))
    # generation-time use of the same arm stays legitimate
    outs = engine.generate(["Say hi."], arm, {"max_new_tokens": 4})
    assert len(outs) == 1


class _NaNArm:
    """A stub handle whose ctx makes the forward emit NaN logits."""

    name, kind, coeff, intervention_hash = "nanarm", "steering", 1.0, "00" * 8
    resolved_spec = {"positions": "all"}

    def __init__(self, model):
        self._model = model

    def ctx(self):
        import contextlib

        def hook(module, args, output):
            output.logits.mul_(float("nan"))  # in place; return None

        @contextlib.contextmanager
        def cm():
            h = self._model.register_forward_hook(hook)
            try:
                yield
            finally:
                h.remove()
        return cm()

    def transform_prompt(self, s):
        return s


def test_score_corpus_refuses_nan_logits(engine, corpus):
    """NaN log-probs make every isinf/κ comparison false, silently
    evading the KL red flags — the engine must refuse the cell
    loudly instead."""
    with pytest.raises(EngineError, match="NaN"):
        list(engine.score_corpus(corpus, _NaNArm(engine.model)))


# ---------------------------------------------------------------------------
# generation contract
# ---------------------------------------------------------------------------


def test_generate_shapes_and_flags(engine):
    outs = engine.generate(["Say hi.", "A much longer prompt about rivers "
                            "and bridges and mills."], None,
                           {"max_new_tokens": 6})
    assert len(outs) == 2
    for o in outs:
        assert isinstance(o.text, str)
        assert 1 <= len(o.gen_token_logprobs) <= 6
        assert all(lp <= 0 for lp in o.gen_token_logprobs)
        assert isinstance(o.ended_with_eos, bool)


def test_generate_refuses_unknown_gen_cfg(engine):
    with pytest.raises(EngineError, match="unknown gen_cfg"):
        engine.generate(["x"], None, {"max_new_tokens": 4, "beam_width": 3})


def test_generate_returns_raw_token_ids(engine):
    """GenOut carries the RAW generated ids (terminal EOS included when
    ended) — the canary computes repetition/length from them, never
    from a re-tokenization of the decoded text."""
    outs = engine.generate(["Say hi.", "Name a color."], None,
                           {"max_new_tokens": 6})
    from sisteer.capeval.serving.hf_engine import _eos_ids
    eos = _eos_ids(engine.tokenizer, engine.model)
    for o in outs:
        assert len(o.gen_token_ids) == len(o.gen_token_logprobs)
        assert all(isinstance(t, int) for t in o.gen_token_ids)
        if o.ended_with_eos:
            assert o.gen_token_ids[-1] in eos


def test_generate_chunks_by_batch_size_with_fresh_ctx_per_chunk(
        engine, vector_artifact, monkeypatch):
    """batch_size chunks the prompt list INSIDE the engine (5 prompts
    @ batch 2 -> 3 model.generate calls, each <= 2 rows) and a FRESH
    arm ctx wraps each chunk — never one giant batch."""
    arm = engine.arm(AblationSpec(name="a", kind="steering",
                                  artifact=str(vector_artifact), layer=1,
                                  coeffs=[2.0]), 2.0)
    batch_rows = []
    real_generate = engine.model.generate

    def spy(*args, **kwargs):
        batch_rows.append(kwargs["input_ids"].shape[0])
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(engine.model, "generate", spy)
    before = arm.ctx_entries
    prompts = [f"Prompt number {i}." for i in range(5)]
    outs = engine.generate(prompts, arm, {"max_new_tokens": 4,
                                          "batch_size": 2})
    assert len(outs) == 5
    assert batch_rows == [2, 2, 1]
    assert arm.ctx_entries - before == 3  # one fresh ctx per chunk
    # no batch_size -> a single chunk, a single ctx
    engine.generate(prompts, arm, {"max_new_tokens": 4})
    assert arm.ctx_entries - before == 4
    assert batch_rows[-1] == 5
    with pytest.raises(EngineError, match="batch_size"):
        engine.generate(prompts, arm, {"batch_size": 0})


def test_generate_greedy_key_and_do_sample_conflict(engine):
    outs = engine.generate(["Say hi."], None,
                           {"max_new_tokens": 4, "greedy": True})
    assert len(outs) == 1
    # greedy=True == do_sample=False: stating both consistently is fine
    engine.generate(["Say hi."], None, {"max_new_tokens": 4, "greedy": True,
                                        "do_sample": False})
    with pytest.raises(EngineError, match="conflicts"):
        engine.generate(["Say hi."], None, {"greedy": True,
                                            "do_sample": True})
    with pytest.raises(EngineError, match="conflicts"):
        engine.generate(["Say hi."], None, {"greedy": False,
                                            "do_sample": False})


def test_generate_chat_key_renders_through_template(engine, monkeypatch):
    from sisteer.capeval.serving import hf_engine as he

    seen = []

    def fake_render(tok, p):
        seen.append(p)
        return f"[chat] {p}"

    monkeypatch.setattr(he, "build_chat_prompt", fake_render)
    engine.generate(["Say hi.", "Name a color."], None,
                    {"max_new_tokens": 4, "chat": True})
    assert seen == ["Say hi.", "Name a color."]
    seen.clear()
    engine.generate(["Say hi."], None, {"max_new_tokens": 4})  # no chat key
    assert seen == []


def test_generate_until_truncates_text_but_keeps_raw_ids(engine):
    [ref] = engine.generate(["Say hi."], None, {"max_new_tokens": 8})
    if not ref.text:
        pytest.skip("tiny model generated empty text — nothing to stop on")
    stop = ref.text[0]
    [out] = engine.generate(["Say hi."], None,
                            {"max_new_tokens": 8, "until": [stop]})
    assert out.text == ""  # truncated at the first stop-string hit
    assert out.gen_token_ids == ref.gen_token_ids  # raw ids stay full
    assert out.gen_token_logprobs == ref.gen_token_logprobs
    with pytest.raises(EngineError, match="until"):
        engine.generate(["Say hi."], None, {"until": "notalist"})


# ---------------------------------------------------------------------------
# arm resolution refusals (kind/layer/coeff validation, §6.1/F9)
# ---------------------------------------------------------------------------


def test_kind_none_is_refused(engine):
    with pytest.raises(EngineError, match="arm=None"):
        engine.arm(AblationSpec(name="n", kind="none"), 0.0)


def test_layer_mismatch_is_refused(engine, vector_artifact):
    with pytest.raises(ManifestError, match="does not match"):
        engine.arm(AblationSpec(name="a", kind="steering",
                                artifact=str(vector_artifact), layer=0,
                                coeffs=[1.0]), 1.0)


def test_kind_object_on_safetensors_is_refused(engine, vector_artifact):
    with pytest.raises(EngineError, match="SteeringVector container"):
        engine.arm(AblationSpec(name="a", kind="sisteer_object",
                                artifact=str(vector_artifact), layer=1,
                                coeffs=[1.0]), 1.0)


def test_zero_coeff_control_requires_zero(engine, vector_artifact):
    with pytest.raises(EngineError, match="coeff"):
        engine.arm(_ctrl("z", "control:zero_coeff",
                         artifact=str(vector_artifact)), 1.0)


def test_missing_artifact_is_refused(engine, tmp_path):
    with pytest.raises(EngineError, match="does not exist"):
        engine.arm(AblationSpec(name="a", kind="steering",
                                artifact=str(tmp_path / "nope.safetensors"),
                                layer=1, coeffs=[1.0]), 1.0)


def test_intervention_hash_moves_with_artifact_and_coeff(engine, tmp_path,
                                                         vector_artifact):
    spec = AblationSpec(name="a", kind="steering",
                        artifact=str(vector_artifact), layer=1, coeffs=[2.0])
    h1 = engine.arm(spec, 2.0).intervention_hash
    h2 = engine.arm(spec, 4.0).intervention_hash  # coeff is a resolved field
    assert h1 != h2 and len(h1) == 16
    sv = SteeringVector(
        vectors={1: torch.randn(16, generator=torch.Generator().manual_seed(9))},
        method="distill_soft", model_name=TINY_LLAMA, instruction_id="test")
    other = tmp_path / "other.safetensors"
    sv.save(other)
    h3 = engine.arm(AblationSpec(name="a", kind="steering",
                                 artifact=str(other), layer=1,
                                 coeffs=[2.0]), 2.0).intervention_hash
    assert h3 != h1  # artifact bytes are hashed (F9)


def test_engine_name_is_hf_and_rows_carry_it(engine):
    """The pairing refusal itself lives in the fingerprint; the
    engine's side of the contract is the stamped name (the tier0 e2e
    asserts every filed row carries engine='hf')."""
    assert engine.name == "hf"
    assert HFEngine.name == "hf"
