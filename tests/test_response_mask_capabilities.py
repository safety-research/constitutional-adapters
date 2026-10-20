"""Response-mask serving on the capabilities leg (2026-08-28).

The defect this pins against: the capabilities leg's only decode-only
serving was the ENGINE-PHASE GATE (``--steer-positions generated``), under
which the prefill — and therefore the FIRST generated token, the decisive
one for a refusal defense — is served by the plain base model, while the
JB legs serve the POSITION MASK (assistant-header prompt tail + every
generated position, token 1 steered). One condition's attack and
capability numbers came from two different interventions, and correct
response semantics were not expressible on this leg at all.

What must hold now:

1. ``positions="response"`` on the engine serves the position mask: the
   forward that emits token 1 IS steered (the decisive assertion — the
   retired gate fails it by construction, pinned as the in-test
   contrast), prompt CONTENT positions stay bit-equal to base, and the
   trailing header positions are inside the mask.
2. H <= 0 refuses loudly, on ANY tokenizer (the collapsed-boundary trap).
3. The resolved spec / intervention hash record the mask — including for
   torch-blob objects, whose mask leg is ``fit_positions`` and which
   would otherwise resolve ``positions: "all"`` and hash identically to
   the unmasked serve of the same artifact.
4. Teacher-forced scoring and control arms refuse the mask
   loudly rather than serving a silently-wrong position set.
5. mmlu under the response mask is scored DEPLOYMENT-EXACT (one steered
   prefill per item; the header LOCATED per request — the mmlu prompt
   carries an "Answer:" assistant prefill after the header, where width
   arithmetic is silently wrong) and reflects the served defense: the
   evidence behind re-including mmlu in the capabilities battery.

CPU, conftest tiny_llama model + a fresh tokenizer copy carrying a
flag-HONORING chat template (H == 4; the stock tiny tokenizer's
Llama-2-style template yields H == 0 and drives the refusal test).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sisteer.capeval.manifest import AblationSpec
from sisteer.capeval.serving import (
    GENERATED_POSITIONS,
    RESPONSE_POSITIONS,
    arm_position_policy,
)
from sisteer.capeval.serving.hf_engine import EngineError, HFEngine
from sisteer.capeval.tiers import tier0 as t0
from sisteer.extraction.psr import PSRObject
from sisteer.models import build_chat_prompt, get_decoder_layers
from sisteer.vectors import SteeringVector

REPO = Path(__file__).resolve().parents[1]
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

STEER_LAYER = 0
OBSERVE_LAYER = 1

#: A template that HONORS add_generation_prompt: the generation render
#: appends "<s>assistant\n" -> H == 4 ('<s>', 'ass', 'istant', '\n'),
#: and derive_assistant_markers locates the same 4 ids.
TEMPLATE = (
    "{% for m in messages %}<s>{{ m['role'] }}\n{{ m['content'] }}</s>"
    "{% endfor %}{% if add_generation_prompt %}<s>assistant\n{% endif %}"
)
GEN_CFG = {"max_new_tokens": 5, "do_sample": False}
USER_TEXTS = ["The quick brown fox jumps over the lazy dog",
              "Salt and flour and water make a simple dough"]


@pytest.fixture(scope="module")
def engine(tiny_llama):
    from transformers import AutoTokenizer

    model, _ = tiny_llama
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    tok.chat_template = TEMPLATE
    eng = HFEngine(TINY_LLAMA, revision=None, preloaded=(model, tok))
    eng.load()
    return eng


@pytest.fixture(scope="module")
def header_len(engine):
    from sisteer.hooks import response_header_len

    h = response_header_len(engine.tokenizer)
    assert h == 4  # the fixture template's contract
    return h


@pytest.fixture(scope="module")
def vector_kwargs(tmp_path_factory):
    d = tmp_path_factory.mktemp("respmask_vec")
    sv = SteeringVector(
        vectors={STEER_LAYER: torch.randn(
            16, generator=torch.Generator().manual_seed(21)) * 4.0},
        method="distill_soft", model_name=TINY_LLAMA,
        instruction_id="respmask")
    path = d / "vec.safetensors"
    sv.save(path)
    return {"name": "vec", "kind": "steering", "artifact": str(path),
            "layer": STEER_LAYER, "coeffs": [1.0]}


@pytest.fixture(scope="module")
def psr_kwargs(tmp_path_factory, tiny_llama):
    model, _tok = tiny_llama
    d = tmp_path_factory.mktemp("respmask_psr")
    g = torch.Generator().manual_seed(22)
    obj = PSRObject(model, [STEER_LAYER])
    obj.set_directions({STEER_LAYER: torch.randn(16, generator=g) * 4.0})
    obj.gate_w[STEER_LAYER].data = 0.05 * torch.randn(16, generator=g)
    path = d / "object.pt"
    obj.save(path)
    return {"name": "psr", "kind": "sisteer_object", "artifact": str(path),
            "layer": STEER_LAYER, "coeffs": [1.0]}


def _arm(engine, kwargs, positions="all", coeff=1.0):
    return engine.arm(AblationSpec(**{**kwargs, "positions": positions}),
                      coeff)


def _prompts(tok):
    return [build_chat_prompt(tok, t) for t in USER_TEXTS]


import contextlib


@contextlib.contextmanager
def _observe(model, layer):
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
# 1. THE DECISIVE ASSERTION: token 1 is steered under the response mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["vector", "psr"])
def test_token1_steered_content_unsteered_header_steered(
        engine, request, which, header_len):
    """The forward that emits the FIRST generated token runs steered under
    positions='response' — and the retired engine-phase gate ('generated')
    fails exactly this assertion (its token 1 is base-identical), pinned
    here as the in-test contrast so the two decode-only mechanisms can
    never be mistaken for each other again."""
    kwargs = request.getfixturevalue(f"{which}_kwargs")
    model, tok = engine.model, engine.tokenizer
    prompts = _prompts(tok)
    arm_resp = _arm(engine, kwargs, positions=RESPONSE_POSITIONS)
    arm_gate = _arm(engine, kwargs, positions=GENERATED_POSITIONS)

    with _observe(model, OBSERVE_LAYER) as base_caps:
        base = engine.generate(prompts, None, GEN_CFG)
    with _observe(model, OBSERVE_LAYER) as resp_caps:
        resp = engine.generate(prompts, arm_resp, GEN_CFG)
    gate = engine.generate(prompts, arm_gate, GEN_CFG)

    # token 1: steered under the response mask (its logits come from the
    # steered final header position of the prefill) ...
    for r, b in zip(resp, base):
        assert (r.gen_token_ids[0] != b.gen_token_ids[0]
                or r.gen_token_logprobs[0] != b.gen_token_logprobs[0]), (
            "token 1 identical to base under the response mask — the "
            "forward that emits it was not steered")
    # ... and base-identical under the retired gate (the contrast: the
    # old decode-only behaviour fails the assertion above).
    for g, b in zip(gate, base):
        assert g.gen_token_ids[0] == b.gen_token_ids[0]
        assert g.gen_token_logprobs[0] == b.gen_token_logprobs[0]

    # prefill footprint: prompt CONTENT bit-equal to base; the trailing
    # H header columns (left padding right-aligns them for every row)
    # inside the mask. Observed one layer downstream of the steer, so
    # causal attention keeps content columns exactly base.
    H = header_len
    assert base_caps[0].shape[1] > H and resp_caps[0].shape[1] > H
    assert torch.equal(resp_caps[0][:, :-H], base_caps[0][:, :-H]), (
        "prompt-content positions were steered — the mask is wider than "
        "header+generated")
    assert not torch.equal(resp_caps[0][:, -H:], base_caps[0][:, -H:]), (
        "the assistant-header positions were NOT steered — the boundary "
        "collapsed to prompt_width (decode-only), the exact defect")


# ---------------------------------------------------------------------------
# 2. H <= 0 refuses loudly, on any tokenizer
# ---------------------------------------------------------------------------


def test_h_zero_refuses_loudly(tiny_llama, vector_kwargs):
    """The stock tiny tokenizer's Llama-2-style template appends nothing
    for add_generation_prompt and carries no known header marker -> H == 0
    -> building a response-mask arm must be a hard error, never a silent
    collapse to full-prompt width."""
    model, tok = tiny_llama
    eng = HFEngine(TINY_LLAMA, revision=None, preloaded=(model, tok))
    eng.load()
    from sisteer.hooks import response_header_len

    assert response_header_len(tok) == 0  # the precondition this pins
    with pytest.raises(EngineError, match="non-zero assistant-header"):
        _arm(eng, vector_kwargs, positions=RESPONSE_POSITIONS)


# ---------------------------------------------------------------------------
# 2b. the gemma-4 shape: the located header absorbs prompt-side scaffold
# ---------------------------------------------------------------------------

#: The GEMMA-4 SHAPE: turns close with scaffold AFTER the terminator
#: ("</s>\n"), so derive_assistant_markers' terminator-anchored header
#: absorbs that newline — located ['\n', '<s>', 'ass', 'istant', '\n']
#: (5) — while add_generation_prompt actually appends only
#: ['<s>', 'ass', 'istant', '\n'] (4). Exactly the pinned
#: google/gemma-4-E4B-it structure (appended 3 vs located 4, verified on
#: revision ee0ef602), reproduced on the tiny tokenizer with no gated
#: download. The leading newline is PROMPT content, present in both
#: renders and inside the fit-time mask's EXCLUDED region, so the fit
#: support is the APPENDED header (hooks.response_header_len == 4 here;
#: gemma-4 artifacts record 3 on their template). Before the fix, the
#: serving guard compared the armed width against the LOCATED length and
#: refused the agreement of two different definitions — the scale-12
#: wave's gemma fitted-cell blocker (Slurm 232411 tasks 4-6).
GEMMA_SHAPE_TEMPLATE = (
    "{% for m in messages %}<s>{{ m['role'] }}\n{{ m['content'] }}</s>\n"
    "{% endfor %}{% if add_generation_prompt %}<s>assistant\n{% endif %}"
)


@pytest.fixture()
def gemma_shape_engine(tiny_llama):
    from transformers import AutoTokenizer

    model, _ = tiny_llama
    tok = AutoTokenizer.from_pretrained(TINY_LLAMA)
    tok.chat_template = GEMMA_SHAPE_TEMPLATE
    eng = HFEngine(TINY_LLAMA, revision=None, preloaded=(model, tok))
    eng.load()
    return eng


def test_gemma_shape_derivations_pin(gemma_shape_engine):
    """The shape this block exists for: width == len(generation ids) ==
    4, and the generation ids are the LOCATED header's tail with the
    prompt-side newline excluded (located len 5)."""
    from sisteer.chat_markers import (derive_assistant_markers,
                                      generation_header_ids)
    from sisteer.hooks import response_header_len

    tok = gemma_shape_engine.tokenizer
    located = derive_assistant_markers(tok).header
    gen = generation_header_ids(tok)
    assert response_header_len(tok) == 4
    assert len(gen) == 4
    assert len(located) == 5
    assert tuple(located[-4:]) == tuple(gen)  # gen ids = located tail


def test_gemma_shape_arms_the_fit_support(gemma_shape_engine,
                                          vector_kwargs, psr_kwargs):
    """Arming positions='response' on the gemma shape SUCCEEDS (the old
    guard refused: 'width-derived header length H=4 disagrees with the
    LOCATED assistant header (len 5)') and the armed mask is the WIDTH —
    the fit-time deriver's value — never the located 5, which would
    steer one prompt-side token outside the fit support."""
    for kwargs in (vector_kwargs, psr_kwargs):
        arm = _arm(gemma_shape_engine, kwargs,
                   positions=RESPONSE_POSITIONS)
        assert arm._obj.response_header_len == 4
        assert arm.resolved_spec["response_header_len"] == 4


def test_gemma_shape_accepts_the_width_fit_object(gemma_shape_engine,
                                                  tiny_llama,
                                                  tmp_path_factory):
    """The staged-artifact situation: a PSR blob whose persisted fit leg
    recorded the width its own fit derived (4 here; gemma-4 artifacts
    record 3 on their template) arms cleanly and serves that SAME
    support. The artifacts are never rewritten."""
    model, _tok = tiny_llama
    d = tmp_path_factory.mktemp("respmask_gemma_fit")
    g = torch.Generator().manual_seed(23)
    obj = PSRObject(model, [STEER_LAYER])
    obj.set_directions({STEER_LAYER: torch.randn(16, generator=g) * 4.0})
    obj.gate_w[STEER_LAYER].data = 0.05 * torch.randn(16, generator=g)
    obj.set_fit_positions("response", 4)  # the width this template derives
    path = d / "object.pt"
    obj.save(path)
    kwargs = {"name": "psr_wfit", "kind": "sisteer_object",
              "artifact": str(path), "layer": STEER_LAYER, "coeffs": [1.0]}
    arm = _arm(gemma_shape_engine, kwargs, positions=RESPONSE_POSITIONS)
    assert arm._obj.response_header_len == 4
    assert arm.resolved_spec["response_header_len"] == 4


def test_recorded_header_from_another_template_refuses(gemma_shape_engine,
                                                       tiny_llama,
                                                       tmp_path_factory):
    """A recorded fit boundary that does not equal this tokenizer's own
    width derivation (like with like — response_header_len is the SAME
    deriver the fit used) is an object fit against a different chat
    template — refused loudly, never served."""
    model, _tok = tiny_llama
    d = tmp_path_factory.mktemp("respmask_drift")
    g = torch.Generator().manual_seed(24)
    obj = PSRObject(model, [STEER_LAYER])
    obj.set_directions({STEER_LAYER: torch.randn(16, generator=g) * 4.0})
    obj.set_fit_positions("response", 9)  # nine-token header: not ours
    path = d / "object.pt"
    obj.save(path)
    kwargs = {"name": "psr_drift", "kind": "sisteer_object",
              "artifact": str(path), "layer": STEER_LAYER, "coeffs": [1.0]}
    with pytest.raises(EngineError, match="different chat template"):
        _arm(gemma_shape_engine, kwargs, positions=RESPONSE_POSITIONS)


# ---------------------------------------------------------------------------
# 3. identity: the mask enters the resolved spec + hash (blobs included)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["vector", "psr"])
def test_resolved_spec_and_hash_record_the_mask(engine, request, which,
                                                header_len):
    kwargs = request.getfixturevalue(f"{which}_kwargs")
    default = _arm(engine, kwargs)
    gate = _arm(engine, kwargs, positions=GENERATED_POSITIONS)
    resp = _arm(engine, kwargs, positions=RESPONSE_POSITIONS)

    # default: the frozen key set — every existing store/hash untouched.
    assert set(default.resolved_spec) == {
        "kind", "object_kind", "layers", "positions", "skip_bos",
        "norm_relative", "coeff", "converter_version"}

    assert resp.resolved_spec["positions"] == RESPONSE_POSITIONS
    assert resp.resolved_spec["response_header_len"] == header_len
    assert arm_position_policy(resp) == RESPONSE_POSITIONS
    # three distinct interventions, three distinct identities — for the
    # blob kind this is exactly the mislabel the explicit override fixes
    # (fit_positions carries the mask; positions would read "all").
    hashes = {default.intervention_hash, gate.intervention_hash,
              resp.intervention_hash}
    assert len(hashes) == 3


# ---------------------------------------------------------------------------
# 4. loud refusals: teacher-forced scoring, control arms
# ---------------------------------------------------------------------------


def test_score_corpus_refuses_the_response_mask(engine, vector_kwargs):
    arm = _arm(engine, vector_kwargs, positions=RESPONSE_POSITIONS)
    corpus = t0.windows_from_texts(
        engine.tokenizer, "respmask",
        ["The river runs east from the high meadow past the mill."],
        window_len=12, min_window_tokens=4)
    with pytest.raises(EngineError, match="response"):
        list(engine.score_corpus(corpus, arm))


def test_control_arm_refuses_response(engine, vector_kwargs):
    with pytest.raises(EngineError, match="control arms do not serve"):
        engine.arm(
            AblationSpec(name="z", kind="control:zero_coeff", seed=7,
                         params={"artifact": vector_kwargs["artifact"]},
                         coeffs=[0.0], positions=RESPONSE_POSITIONS), 0.0)


# ---------------------------------------------------------------------------
# 5. mmlu under the response mask: deployment-exact, and it measures the
#    served defense (the re-inclusion evidence)
# ---------------------------------------------------------------------------

MMLU_ROWS = [
    {"item_id": "m0000-dddddddd", "question": "Pick B.",
     "choices": ["one", "two", "three", "four"], "answer": 1},
    {"item_id": "m0001-eeeeeeee", "question": "Pick D.",
     "choices": ["ay", "bee", "sea", "dee"], "answer": 3},
]


def _mmlu_ids_and_letters(engine, arm):
    from sisteer.capeval.tiers import parity
    from sisteer.eval.capability_ext import _letter_token_ids, _letters
    from sisteer.generate import encode_kwargs

    tok = engine.tokenizer
    prompts_ids, letter_ids = [], []
    for row in MMLU_ROWS:
        prompt = parity._mmlu_prompt(tok, row)
        if arm is not None:
            prompt = arm.transform_prompt(prompt)
        ids = list(tok(prompt, **encode_kwargs(tok, prompt))["input_ids"])
        letters = _letters(min(len(row["choices"]), 4))
        prompts_ids.append(ids)
        letter_ids.append([int(l) for l in _letter_token_ids(tok, letters)])
    return prompts_ids, letter_ids


def test_mmlu_response_scoring_reflects_the_served_defense(
        engine, vector_kwargs, header_len):
    """Does the steered header row actually shape the mmlu_letter readout?
    Yes — verified here concretely: the deployment-exact letter NLLs under
    the response-masked arm differ from the arm-free NLLs on every item
    (under the retired gate they were base-identical by construction,
    which is why mmlu was excluded). Also pins that the boundary is
    LOCATED: the mmlu prompt ends with an 'Answer:' assistant PREFILL, so
    the header ids sit strictly BEFORE the prompt tail — width arithmetic
    (prompt_width − H) would mask the wrong positions — and that the
    armed boundary is restored after scoring."""
    from sisteer.capeval.tiers import parity
    from sisteer.chat_markers import derive_assistant_markers

    arm = _arm(engine, vector_kwargs, positions=RESPONSE_POSITIONS)
    prompts_ids, letter_ids = _mmlu_ids_and_letters(engine, arm)

    # the assistant-prefill fact: the located header is NOT the prompt tail
    markers = derive_assistant_markers(engine.tokenizer)
    for ids in prompts_ids:
        start = parity._rfind_subseq(ids, list(markers.header))
        assert start >= 0
        assert start < len(ids) - len(markers.header), (
            "the mmlu prompt should carry the 'Answer:' prefill AFTER the "
            "header — otherwise this test does not exercise the located "
            "boundary")

    resp = parity._response_letter_nlls(engine, arm, prompts_ids, letter_ids)

    # arm-free reference: the same forward outside any ctx
    model = engine.model
    base = []
    for ids, lids in zip(prompts_ids, letter_ids):
        t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=t,
                           attention_mask=torch.ones_like(t)).logits
        lp = torch.log_softmax(logits[0, -1, :].double(), dim=-1)
        base.append([-float(lp[l]) for l in lids])

    for r, b in zip(resp, base):
        assert any(abs(x - y) > 1e-9 for x, y in zip(r, b)), (
            "response-masked letter NLLs identical to base — the letter "
            "readout is NOT being shaped by the steered header row")

    # the armed boundary survives the per-request located overrides
    assert arm._obj.response_header_len == header_len


def test_mmlu_response_scoring_runs_through_parity_run(
        tmp_path, engine, vector_kwargs):
    """End to end through parity.run: the response-policy arm scores mmlu
    (n == len(rows), per-item rows emitted) instead of refusing or
    silently scoring arm-free."""
    from sisteer.capeval import store as cs
    from sisteer.capeval.tiers import parity

    arm = _arm(engine, vector_kwargs, positions=RESPONSE_POSITIONS)
    template = {
        "sweep": "respmask_test", "run_id": "r0", "model_id": TINY_LLAMA,
        "model_revision": None, "arm_name": "vec", "kind": "steering",
        "coeff": 1.0, "intervention_hash": arm.intervention_hash,
        "engine": "hf", "trial_id": 1, "seed": 1, "attempt": 0,
        "harness_config_hash": "0f1e2d3c4b5a6978",
        "env_digest": "99aabbccddeeff00", "slurm_job_id": None,
    }
    store = cs.CellStore(
        tmp_path, "vec__tier01__t1__a0__a1b2c3d4__0f1e2d3c",
        row_constants=template)
    spec = parity.ParitySpec(
        template=template,
        domain_of={parity.PARITY_TASK_GSM8K: "math_s2parity",
                   parity.PARITY_TASK_MMLU: "knowledge_s2parity"},
        # PW3: the HF driver's shape — the retired gsm8k probe is not run
        # (empty rows), the mmlu probe runs alone
        gsm8k_rows=[],
        mmlu_rows=list(MMLU_ROWS), order_seed=1,
        gen={"max_new_tokens": 4, "do_sample": False, "batch_size": 8})
    summary = parity.run(engine, arm, spec, store)
    assert summary["mmlu"]["n"] == len(MMLU_ROWS)
    assert "gsm8k" not in summary


def test_mmlu_response_refuses_headerless_prompt(engine, vector_kwargs):
    """A prompt whose tokens do not contain the located header ids must
    refuse — the boundary is located in the request's own tokens, never
    assumed."""
    from sisteer.capeval.tiers import parity

    arm = _arm(engine, vector_kwargs, positions=RESPONSE_POSITIONS)
    with pytest.raises(parity.ParityError, match="do not occur"):
        parity._response_letter_nlls(engine, arm, [[5, 6, 7, 8]], [[1]])


# ---------------------------------------------------------------------------
# 6. the seam vocabulary: 'generated' refused, never aliased
# ---------------------------------------------------------------------------


def test_arm_spec_for_cell_vocabulary(tmp_path):
    from types import SimpleNamespace

    from sisteer.eval.cell_controls import arm_spec_for_cell

    obj = tmp_path / "object.pt"
    obj.write_bytes(b"x")
    cell = SimpleNamespace(id="safety_refusal__psr_kl_all", arm="psr_kl_all",
                           si_id="safety_refusal", object_path=obj,
                           stacked=False)
    base_cfg = {"layer": 15, "scale": 1.0}

    spec = arm_spec_for_cell(cell, {**base_cfg, "steer_positions": "response"})
    assert spec["positions"] == "response"
    spec = arm_spec_for_cell(cell, dict(base_cfg))
    assert "positions" not in spec  # default: byte-identical spec/hash
    with pytest.raises(ValueError, match="response"):
        arm_spec_for_cell(cell, {**base_cfg, "steer_positions": "generated"})
    with pytest.raises(ValueError, match="unknown steer_positions"):
        arm_spec_for_cell(cell, {**base_cfg, "steer_positions": "decode"})
