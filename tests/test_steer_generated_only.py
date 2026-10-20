"""Generated-only steering policy (engine seam, 2026-08-20).

``positions="generated"`` on a manifested arm asks the ENGINE — not the
object — to serve the intervention on generated positions only: the
prefill forward runs OUTSIDE the arm ctx (the plain base model encodes
the prompt, so every prompt position and the KV entries decode attends
to are unsteered, and the first generated token is picked from the
unsteered prompt logits), and every DECODE forward runs INSIDE one fresh
ctx. Because every registered arm applies through the same enterable
ctx (hooks for steering/PSR-style objects, weight edits for weight
arms), gating at ctx enter/exit is method-generic by construction —
these tests therefore run the SAME assertions over three different
method kinds (a safetensors steering vector, a torch-blob PSR object,
and a rank-one LoRA weight adapter — for which the mode means "prompt
encoded by the base model, generation computed under the adapter"):

(a) prompt-position hidden states under the mode are bit-equal to the
    base model's (and the first generated token/logprob matches base);
(b) generated positions ARE steered (decode-step states/logprobs
    differ from base on identical inputs);
(c) mode off is byte-identical to the existing serving, and a
    zero-effect arm under the mode is byte-identical to the base model
    (the two-phase stitch alone changes nothing);
plus the policy plumbing: resolved-spec/hash additivity, the
score_corpus refusal, the F10 one-ctx-per-chunk
discipline, and the driver's --steer-positions seam.

Semantics this implementation pins (to be consumed VERBATIM by the
shared train/test subset-steering primitive that will live
in src/sisteer/hooks.py — the train{all,decode-only} x
test{all,decode-only} 2x2):

- FIRST GENERATED TOKEN: sampled from the UNSTEERED prompt logits (the
  prefill picks it), but its POSITION is steered from the first decode
  forward on — the intervention shapes its residual processing and
  every later token, never its own sampling distribution.
- CHAT-HEADER SPECIALS: everything rendered into the prompt (the
  assistant header included, via add_generation_prompt) is prompt =
  unsteered; any special token the model EMITS mid-generation is a
  generated position and is steered like any other.
- NO-OP-ON-PREFILL, not runs-and-writes-nothing: the arm ctx is never
  entered for the prefill, so the arm machinery is not invoked at any
  prompt position. Behaviorally identical for the forward itself, but
  DIFFERENT for methods that stash pre-steering activations while
  hooked (e.g. PSRObject._last_gate) — a runs-and-writes-nothing
  spelling would still populate such stashes over prompt positions.

FLAGGED FUTURE TEST (document now, build when the shared masking
primitive lands): a CPU fixture test asserting the TRAIN-time
decode-only mask and THIS test-time policy select IDENTICAL
boolean position sets on a fixed batch (same tokenizer, same padding,
same chat template). It gates the 2x2's off-diagonal cells — without
it, "train decode-only, test decode-only" can silently mean two
different position sets.

CPU, conftest tiny_llama (float32).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest
import torch

from sisteer.capeval.manifest import AblationSpec
from sisteer.capeval.serving import GENERATED_POSITIONS, arm_position_policy
from sisteer.capeval.serving.hf_engine import EngineError, HFEngine
from sisteer.capeval.tiers import tier0 as t0
from sisteer.extraction.objects import RankOneLoRAObject
from sisteer.extraction.psr import PSRObject
from sisteer.models import get_decoder_layers
from sisteer.vectors import SteeringVector

REPO = Path(__file__).resolve().parents[1]
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # conftest pin

#: steer layer 0, observe layer 1: a test capture hook registers BEFORE the
#: arm ctx's own hooks, so on the steered layer itself it sees PRE-steering
#: values — one layer downstream the steering is visible unambiguously.
STEER_LAYER = 0
OBSERVE_LAYER = 1

PROMPTS = ["The quick brown fox jumps over the lazy dog",
           "Salt and flour and water make a simple dough"]
GEN_CFG = {"max_new_tokens": 6, "do_sample": False}


@pytest.fixture(scope="module")
def engine(tiny_llama):
    model, tok = tiny_llama
    eng = HFEngine(TINY_LLAMA, revision=None, preloaded=(model, tok))
    eng.load()
    return eng


@pytest.fixture(scope="module")
def vector_kwargs(tmp_path_factory):
    d = tmp_path_factory.mktemp("genonly_vec")
    sv = SteeringVector(
        vectors={STEER_LAYER: torch.randn(
            16, generator=torch.Generator().manual_seed(11)) * 4.0},
        method="distill_soft", model_name=TINY_LLAMA, instruction_id="genonly")
    path = d / "vec.safetensors"
    sv.save(path)
    return {"name": "vec", "kind": "steering", "artifact": str(path),
            "layer": STEER_LAYER, "coeffs": [1.0]}


@pytest.fixture(scope="module")
def psr_kwargs(tmp_path_factory, tiny_llama):
    model, _tok = tiny_llama
    d = tmp_path_factory.mktemp("genonly_psr")
    g = torch.Generator().manual_seed(12)
    obj = PSRObject(model, [STEER_LAYER])
    obj.set_directions({STEER_LAYER: torch.randn(16, generator=g) * 4.0})
    obj.gate_w[STEER_LAYER].data = 0.05 * torch.randn(16, generator=g)
    path = d / "object.pt"
    obj.save(path)
    return {"name": "psr", "kind": "sisteer_object", "artifact": str(path),
            "layer": STEER_LAYER, "coeffs": [1.0]}


@pytest.fixture(scope="module")
def lora_kwargs(tmp_path_factory, tiny_llama):
    """The weight-adapter kind (rank-one LoRA on o_proj): under the mode
    the prompt is encoded by the BASE model and only generated positions
    compute under the adapter — the strongest genericity check, since
    its ctx is a weight-path edit, not a residual hook."""
    model, _tok = tiny_llama
    d = tmp_path_factory.mktemp("genonly_lora")
    g = torch.Generator().manual_seed(13)
    obj = RankOneLoRAObject(model, layer=STEER_LAYER)
    obj.B.data = torch.randn(obj.B.shape[0], generator=g) * 2.0
    path = d / "object.pt"
    obj.save(path)
    return {"name": "lora", "kind": "sisteer_object", "artifact": str(path),
            "layer": STEER_LAYER, "coeffs": [1.0]}


def _arm(engine, kwargs, positions="all", coeff=1.0):
    return engine.arm(AblationSpec(**{**kwargs, "positions": positions}),
                      coeff)


def _spec_kwargs(request, which):
    return request.getfixturevalue(f"{which}_kwargs")


@contextlib.contextmanager
def _observe(model, layer):
    """Capture the [B, S, d] outputs of one decoder layer, one entry per
    forward, registered BEFORE any arm ctx (so the steered layer itself
    would show pre-steering values — observe downstream instead)."""
    captured = []
    mod = get_decoder_layers(model)[layer]

    def hook(_m, _a, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().clone())

    handle = mod.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# (a)+(b): prompt positions unsteered, generated positions steered —
# the SAME assertions for both method kinds (genericity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["vector", "psr", "lora"])
def test_prompt_unsteered_generation_steered(engine, request, which):
    kwargs = _spec_kwargs(request, which)
    model = engine.model
    arm_gen = _arm(engine, kwargs, positions=GENERATED_POSITIONS)
    arm_all = _arm(engine, kwargs)

    with _observe(model, OBSERVE_LAYER) as base_caps:
        base = engine.generate(PROMPTS, None, GEN_CFG)
    with _observe(model, OBSERVE_LAYER) as gen_caps:
        gen = engine.generate(PROMPTS, arm_gen, GEN_CFG)
    with _observe(model, OBSERVE_LAYER) as all_caps:
        alls = engine.generate(PROMPTS, arm_all, GEN_CFG)

    # capture 0 is the prefill forward ([B, in_len, d]); the rest are
    # single-position decode forwards.
    assert base_caps[0].shape[1] > 1 and gen_caps[0].shape[1] > 1

    # (a) prompt-position hidden states: BIT-equal to base under the
    # generated-only policy; different under the existing all-positions
    # serving (the contrast that shows prompt steering was removed).
    assert torch.equal(gen_caps[0], base_caps[0])
    assert not torch.equal(all_caps[0], base_caps[0])

    # (a) and therefore the first generated token and its logprob come
    # from the unsteered prompt distribution — identical to base.
    for g, b in zip(gen, base):
        assert g.gen_token_ids[0] == b.gen_token_ids[0]
        assert g.gen_token_logprobs[0] == b.gen_token_logprobs[0]

    # (b) generated positions ARE steered: the first decode forward
    # consumes the SAME token as base (just asserted), so any state
    # difference is the intervention acting on a generated position.
    assert not torch.equal(gen_caps[1], base_caps[1])
    assert any(g.gen_token_logprobs[1] != b.gen_token_logprobs[1]
               for g, b in zip(gen, base)
               if len(g.gen_token_logprobs) > 1
               and len(b.gen_token_logprobs) > 1)


# ---------------------------------------------------------------------------
# (c) default-preserving: mode off == the existing serving, byte-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["vector", "psr", "lora"])
def test_mode_off_is_byte_identical_to_existing_serving(
        engine, request, which):
    """The default path must still be EXACTLY the pre-change serving: one
    model.generate call wrapped in one arm ctx. The reference below IS
    that serving, built from the same primitives."""
    from sisteer.generate import encode_kwargs

    kwargs = _spec_kwargs(request, which)
    tok, model = engine.tokenizer, engine.model
    prompt = PROMPTS[0]
    arm = _arm(engine, kwargs)

    out = engine.generate([prompt], arm, GEN_CFG)[0]

    ids = torch.tensor([tok(prompt, **encode_kwargs(tok, prompt))["input_ids"]])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad(), arm.ctx():
        ref = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=GEN_CFG["max_new_tokens"], do_sample=False,
            pad_token_id=pad_id, return_dict_in_generate=True,
            output_scores=True)
    ref_tokens = ref.sequences[0, ids.shape[1]:]
    n = len(out.gen_token_ids)  # engine truncates at the first EOS
    assert out.gen_token_ids == [int(t) for t in ref_tokens[:n]]
    for step in range(n):
        lp = torch.log_softmax(ref.scores[step][0].double(), dim=-1)
        assert out.gen_token_logprobs[step] == float(lp[ref_tokens[step]])


def test_zero_effect_arm_under_mode_is_byte_identical_to_base(
        engine, vector_kwargs):
    """The two-phase stitch (prefill + 1 token, then continuation from the
    returned KV cache) must contribute NOTHING by itself: a zero-coeff
    control arm under the generated policy reproduces the base model
    bit-exactly, token ids and logprobs both."""
    arm = engine.arm(
        AblationSpec(name="z", kind="control:zero_coeff", seed=7,
                     params={"artifact": vector_kwargs["artifact"]},
                     coeffs=[0.0], positions=GENERATED_POSITIONS), 0.0)
    base = engine.generate(PROMPTS, None, GEN_CFG)
    zero = engine.generate(PROMPTS, arm, GEN_CFG)
    assert arm.ctx_entries == 1  # the decode phase entered the ctx
    for z, b in zip(zero, base):
        assert z.gen_token_ids == b.gen_token_ids
        assert z.gen_token_logprobs == b.gen_token_logprobs
        assert z.ended_with_eos == b.ended_with_eos


# ---------------------------------------------------------------------------
# identity plumbing: resolved spec / hash additivity, position policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["vector", "psr", "lora"])
def test_resolved_spec_and_hash_are_additive(engine, request, which):
    kwargs = _spec_kwargs(request, which)
    default = _arm(engine, kwargs)
    gen = _arm(engine, kwargs, positions=GENERATED_POSITIONS)

    # default: the frozen key set, no new key, policy "all" — every
    # existing store/hash is untouched.
    assert set(default.resolved_spec) == {
        "kind", "object_kind", "layers", "positions", "skip_bos",
        "norm_relative", "coeff", "converter_version"}
    assert arm_position_policy(default) == "all"

    # generated: the key appears, the OBJECT's own position mode stays
    # its default (the engine gates ctx entry instead), and the
    # intervention identity separates from the default arm's.
    assert gen.resolved_spec["steer_positions"] == GENERATED_POSITIONS
    assert gen.resolved_spec["positions"] == "all"
    assert arm_position_policy(gen) == GENERATED_POSITIONS
    assert gen.intervention_hash != default.intervention_hash


def test_score_corpus_refuses_the_generated_policy(engine, vector_kwargs):
    """Teacher-forced scoring has NO generated positions, so it must refuse
    loudly rather than silently serving the wrong policy."""
    arm = _arm(engine, vector_kwargs, positions=GENERATED_POSITIONS)
    corpus = t0.windows_from_texts(
        engine.tokenizer, "genonly",
        ["The river runs east from the high meadow past the mill."],
        window_len=12, min_window_tokens=4)
    with pytest.raises(EngineError, match="generated"):
        list(engine.score_corpus(corpus, arm))


@pytest.mark.parametrize("which", ["psr", "lora"])
def test_prefill_forward_runs_entirely_outside_the_arm_ctx(
        engine, request, which):
    """The SHARP distinction (pinned cross-path with the vLLM proxy lane,
    branch integrate-main, 2026-08-20): decode-only means the arm
    machinery is NOT INVOKED during the prefill — never merely
    invoked-with-zero-effect. For a purely additive op the two spellings
    are byte-indistinguishable, so this asserts on the GATED object
    (PSR's relu gate reads the hidden state) and the weight-edit kind:
    every model forward that runs INSIDE the arm ctx is decode-shaped
    (one position), and the prompt-shaped prefill forward happens outside
    it. If the implementation ever drifts to a scale-0-style spelling of
    the prefill, this is the test that catches it."""
    kwargs = _spec_kwargs(request, which)
    arm = _arm(engine, kwargs, positions=GENERATED_POSITIONS)

    in_ctx_lens = []
    real_ctx = arm.ctx

    def spying_ctx():
        inner = real_ctx()

        @contextlib.contextmanager
        def wrapped():
            def spy(_m, args, kw):
                ids = kw.get("input_ids", args[0] if args else None)
                if ids is not None:
                    in_ctx_lens.append(int(ids.shape[1]))
                return None

            h = engine.model.register_forward_pre_hook(spy, with_kwargs=True)
            try:
                with inner:
                    yield
            finally:
                h.remove()

        return wrapped()

    arm.ctx = spying_ctx
    engine.generate(PROMPTS, arm, GEN_CFG)
    assert in_ctx_lens, "the decode phase entered the arm ctx"
    assert all(s == 1 for s in in_ctx_lens)  # decode-shaped forwards only
    assert len(in_ctx_lens) <= GEN_CFG["max_new_tokens"] - 1


def test_one_ctx_per_chunk_prefill_outside(engine, vector_kwargs):
    """F10 discipline holds under the mode: ONE fresh ctx per generate
    chunk (entered for the decode phase; the prefill runs outside any
    ctx — pinned bit-exactly by the prompt-unsteered test above)."""
    arm = _arm(engine, vector_kwargs, positions=GENERATED_POSITIONS)
    assert arm.ctx_entries == 0
    engine.generate(PROMPTS, arm, GEN_CFG)
    assert arm.ctx_entries == 1
    engine.generate(PROMPTS, arm, {**GEN_CFG, "batch_size": 1})
    assert arm.ctx_entries == 3  # two single-prompt chunks


# ---------------------------------------------------------------------------
# driver seam: --steer-positions reaches the spec + the resolved config
# ---------------------------------------------------------------------------


def test_steer_positions_seam_reaches_spec_and_resolved_config(tmp_path):
    """Mirrors the --coeff seam test: cfg['steer_positions'] must reach
    (a) the AblationSpec positions the engine arms, and (b) the
    generate-time resolved config — and BOTH must stay byte-identical to
    the pre-seam world at the default (the key absent, positions 'all'),
    so existing stores keep their harness_config_hash."""
    from types import SimpleNamespace

    import yaml

    sys.path.insert(0, str(REPO / "scripts"))
    import cell_controls_run as run
    from sisteer.eval.cell_controls import cell_serving, registry_for_controls
    from sisteer.eval.defense_cell import by_id

    ddir = tmp_path / "cells"
    (ddir / "safety_refusal__psr_kl_all").mkdir(parents=True)
    (ddir / "safety_refusal__psr_kl_all" / "object.pt").write_bytes(b"x")
    base_cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())

    captured = {}

    class FakeEngine:
        def arm(self, spec, coeff):
            captured["spec"] = spec
            return "handle"

    # "response" is the decode-only literal since 2026-08-28; "generated"
    # (the retired engine-phase gate) is REFUSED at this seam, never
    # aliased — pinned below.
    for mode, want_positions in (("response", "response"), (None, "all")):
        cfg = dict(base_cfg)
        if mode:
            cfg["steer_positions"] = mode
        reg = by_id(registry_for_controls(cfg, repo_root=REPO,
                                          defenses_dir=ddir))
        cell = reg["safety_refusal__psr_kl_all"]
        handle, system = cell_serving(cell, cfg, FakeEngine())
        assert handle == "handle" and system is None
        assert captured["spec"].positions == want_positions

        resolved = run._resolved_config(SimpleNamespace(id=cell.id), cfg,
                                        None, 0, "claude-haiku-4-5")
        if mode:
            assert resolved["steer_positions"] == "response"
        else:
            assert "steer_positions" not in resolved

    cfg = dict(base_cfg)
    cfg["steer_positions"] = "generated"
    reg = by_id(registry_for_controls(cfg, repo_root=REPO,
                                      defenses_dir=ddir))
    with pytest.raises(ValueError, match="response"):
        cell_serving(reg["safety_refusal__psr_kl_all"], cfg, FakeEngine())
