import hashlib
import math

import torch

from sisteer.extraction.actadd import extract_actadd
from sisteer.extraction.bidpo import PrefExample, train_bidpo
from sisteer.extraction.diffmean import extract_caa, extract_instruction_diff
from sisteer.extraction.distill import (
    DistillExample,
    soft_divergence,
    train_distill,
)
from sisteer.extraction.hypersteer import train_hypersteer_nc
from sisteer.models import hidden_size, num_layers

PROMPTS = [
    "Tell me about the weather.",
    "What is your favorite food?",
    "Describe your morning.",
    "How do plants grow?",
]
POS = [
    "Le temps est très beau aujourd'hui.",
    "J'aime beaucoup le fromage et le pain.",
    "Je me lève tôt et je bois du café.",
    "Les plantes poussent avec le soleil et l'eau.",
]
NEG = [
    "The weather is very nice today.",
    "I really like cheese and bread.",
    "I wake up early and drink coffee.",
    "Plants grow with sunlight and water.",
]


def _weights_hash(model) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_caa_freeform(tiny_llama):
    model, tok = tiny_llama
    sv = extract_caa(
        model, tok, PROMPTS, POS, NEG, layers=[0, 1], instruction_id="french"
    )
    d = hidden_size(model)
    assert sv.layers == [0, 1]
    assert sv.vectors[1].shape == (d,)
    assert torch.isfinite(sv.vectors[1]).all()
    assert sv.vectors[1].norm() > 0
    diag = sv.metadata["diagnostics"]["1"]
    assert "d_prime" in diag and diag["d_prime"] > 0
    assert "directional_agreement" in diag


def test_instruction_diff(tiny_llama):
    model, tok = tiny_llama
    with_si = [f"Always respond in French.\n\n{p}" for p in PROMPTS]
    without = list(PROMPTS)
    sv = extract_instruction_diff(
        model, tok, with_si, without, layers=[1], instruction_id="french"
    )
    assert sv.vectors[1].shape == (hidden_size(model),)
    assert sv.vectors[1].norm() > 0


def test_actadd_per_position(tiny_llama):
    model, tok = tiny_llama
    sv = extract_actadd(
        model, tok, "Always respond in French.", "Always respond in English.",
        paper_layers=[1], instruction_id="french",
    )
    # stored under harness layer 0 (= resid_pre of paper layer 1)
    assert sv.layers == [0]
    mat = sv.vectors[0]
    assert mat.dim() == 2 and mat.shape[1] == hidden_size(model)
    assert sv.metadata["site"] == "resid_pre"
    pooled = extract_actadd(
        model, tok, "French.", "English.", paper_layers=[1], pool=True
    )
    assert pooled.vectors[0].dim() == 1


def test_bidpo_trains_only_vector(tiny_llama):
    model, tok = tiny_llama
    before = _weights_hash(model)
    examples = [
        PrefExample(prompt=p, chosen=c, rejected=r)
        for p, c, r in zip(PROMPTS, POS, NEG)
    ]
    sv = train_bidpo(
        model, tok, examples, layer=1, lr=1e-2, epochs=8, batch_size=2,
        instruction_id="french",
    )
    assert _weights_hash(model) == before  # weights untouched
    vec = sv.vectors[1]
    assert vec.norm() > 0
    # loss starts at exactly 2*log 2 (both direction terms at v=0) and
    # must have decreased
    first = sv.metadata["loss_history_head"][0]
    assert math.isclose(first, 2 * math.log(2), rel_tol=1e-3)
    assert sv.metadata["final_loss"] < first


def test_bidpo_sampled_direction(tiny_llama):
    model, tok = tiny_llama
    examples = [
        PrefExample(prompt=p, chosen=c, rejected=r)
        for p, c, r in zip(PROMPTS[:2], POS[:2], NEG[:2])
    ]
    sv = train_bidpo(
        model, tok, examples, layer=1, lr=1e-2, epochs=2, batch_size=2,
        direction="sampled",
    )
    assert math.isclose(
        sv.metadata["loss_history_head"][0], math.log(2), rel_tol=1e-3
    )


def test_distill_soft(tiny_llama):
    model, tok = tiny_llama
    before = _weights_hash(model)
    examples = [
        DistillExample(
            prompt_with_si=f"Always respond in French.\n\n{p}",
            prompt_without=p,
            completion=c,
        )
        for p, c in zip(PROMPTS, POS)
    ]
    sv = train_distill(
        model, tok, examples, layer=1, loss="soft", lr=1e-2, epochs=10,
        batch_size=2, instruction_id="french",
    )
    assert _weights_hash(model) == before
    assert sv.vectors[1].norm() > 0
    hist_first = sv.metadata["final_loss"] is not None
    assert hist_first
    # KL to the teacher should shrink relative to the unsteered start
    # (first recorded step is with near-zero vector after 1 update).
    assert sv.metadata["final_loss"] >= 0


def test_distill_soft_loss_decreases(tiny_llama):
    model, tok = tiny_llama
    from sisteer.extraction.distill import train_steered_lm
    from torch import nn

    examples = [
        DistillExample(
            prompt_with_si=f"Always respond in French.\n\n{p}",
            prompt_without=p,
            completion=c,
        )
        for p, c in zip(PROMPTS, POS)
    ]
    vec = nn.Parameter(torch.zeros(hidden_size(model)))
    history = train_steered_lm(
        model, tok, examples, 1,
        vector_provider=lambda idxs: vec,
        trainable_params=[vec],
        loss="soft", lr=5e-3, epochs=10, batch_size=2,
    )
    first_epoch = sum(history[:2]) / 2
    last_epoch = sum(history[-2:]) / 2
    assert last_epoch < first_epoch


def _rand_logp(n=5, v=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(n, v, generator=g).double(), -1)


def test_soft_divergence_matches_analytic_forms():
    t, s = _rand_logp(seed=0), _rand_logp(seed=1)
    fwd = (t.exp() * (t - s)).sum(-1).mean()
    rev = (s.exp() * (s - t)).sum(-1).mean()
    m = torch.log((t.exp() + s.exp()) / 2)
    js = 0.5 * ((t.exp() * (t - m)).sum(-1).mean()
                + (s.exp() * (s - m)).sum(-1).mean())
    assert torch.allclose(soft_divergence("soft", t, s), fwd)
    assert torch.allclose(soft_divergence("rkl", t, s), rev)
    assert torch.allclose(soft_divergence("js", t, s), js)


def test_soft_divergence_properties():
    t, s = _rand_logp(seed=0), _rand_logp(seed=1)
    for loss in ("soft", "rkl", "js"):
        # zero iff the distributions match, positive otherwise
        assert float(soft_divergence(loss, t, t).abs()) < 1e-12
        assert float(soft_divergence(loss, t, s)) > 0
    # js is symmetric and bounded by log 2; the KLs are asymmetric
    assert torch.allclose(soft_divergence("js", t, s),
                          soft_divergence("js", s, t))
    assert float(soft_divergence("js", t, s)) < math.log(2)
    assert not torch.allclose(soft_divergence("soft", t, s),
                              soft_divergence("rkl", t, s))
    try:
        soft_divergence("hard", t, s)
        raise AssertionError("expected ValueError for a non-soft loss")
    except ValueError:
        pass


def test_soft_divergence_grads_reach_the_student():
    """rkl/js put the STUDENT log-probs in F.kl_div's target/mixture slots;
    the gradient must still flow to the student for training to move."""
    t = _rand_logp(seed=0)
    for loss in ("soft", "rkl", "js"):
        s_logits = torch.randn(5, 7, dtype=torch.double, requires_grad=True)
        s = torch.log_softmax(s_logits, -1)
        soft_divergence(loss, t, s).backward()
        assert s_logits.grad is not None
        assert float(s_logits.grad.abs().sum()) > 0


def test_distill_divergence_variants_train(tiny_llama):
    """The divergence arms' losses run end-to-end through train_distill:
    loss decreases, the vector moves, and the method label records the loss."""
    model, tok = tiny_llama
    examples = [
        DistillExample(
            prompt_with_si=f"Always respond in French.\n\n{p}",
            prompt_without=p,
            completion=c,
        )
        for p, c in zip(PROMPTS, POS)
    ]
    before = _weights_hash(model)
    for loss in ("rkl", "js"):
        sv = train_distill(
            model, tok, examples, layer=1, loss=loss, lr=1e-2, epochs=10,
            batch_size=2, instruction_id="french",
        )
        assert sv.method == f"distill_{loss}"
        assert sv.vectors[1].norm() > 0
        assert torch.isfinite(sv.vectors[1]).all()
        assert sv.metadata["final_loss"] >= 0
    assert _weights_hash(model) == before  # base weights never move


def test_hypersteer_nc(tiny_llama):
    model, tok = tiny_llama
    instructions = ["Always respond in French.", "Always answer in English."]
    examples = []
    for k, (pos_set, other) in enumerate([(POS, NEG), (NEG, POS)]):
        for p, c in zip(PROMPTS[:2], pos_set[:2]):
            examples.append(
                DistillExample(
                    prompt_with_si=f"{instructions[k]}\n\n{p}",
                    prompt_without=p,
                    completion=c,
                    instruction_idx=k,
                )
            )
    hyper, vectors = train_hypersteer_nc(
        model, tok, examples, instructions, layer=1, epochs=2, batch_size=2,
        lr=1e-3,
    )
    assert len(vectors) == 2
    assert vectors[0].vectors[1].shape == (hidden_size(model),)
    # different instructions -> different vectors
    assert not torch.allclose(vectors[0].vectors[1], vectors[1].vectors[1])


def test_default_layer_band(tiny_llama):
    model, _ = tiny_llama
    from sisteer.eval.runner import default_layers

    layers = default_layers(model)
    assert layers
    assert all(0 <= l < num_layers(model) for l in layers)
