"""loss_impl="chunked_ce": the chunked LM-head cross-entropy path of
train_object_ce equals the full-logits path — loss AND gradients — on a
tiny CPU fp32 model with random adapters, replicates a model's final logit
softcap when its config declares one, records itself as a fit-identity
dimension, and bounds a run with max_steps without touching the objective.

CPU-only, tiny models (the cached tiny-random Llama; a from-config tiny
Gemma2 for the softcap), synthetic doc-shaped examples.
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sisteer.data.pairs import Pair, docs_to_ce_examples
from sisteer.extraction import chunked_ce as cc
from sisteer.extraction.distill import (_encode_aligned,
                                        _gather_completion_logits)
from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.extraction.objects import train_object_ce
from sisteer.extraction.psr import PSRObject
from sisteer.extraction.traindata import collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

SI = "constv2-mix-1k"


def _docs(n=6):
    rows = [
        ("Sys scaffold\n\n", "A short doc."),
        ("", "A much longer story body that runs on for quite a few more "
             "tokens than the short one does, on purpose, and then some."),
        ("Another sys\n\n", "Medium length advice body follows here."),
        ("", "Tiny."),
        ("Third sys with a deliberately long masked prefix that occupies "
         "many positions before any target appears\n\n", "Short target."),
        ("", "Final story text, middling length overall, for the record."),
    ]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id=SI)
            for p, pos in rows[:n]]


def _examples(tok, n=6):
    return docs_to_ce_examples(tok, _docs(n))


def _fresh(kind: str, model):
    if kind == "psr":
        return PSRObject(model, [0, 1], train_direction=True)
    return MultiLayerLoRAObject(model, rank=4)


def _params(obj):
    return [p.detach().clone() for p in obj.trainable]


# -- (a) chunked == full: loss and gradients ----------------------------------


@pytest.mark.parametrize("kind", ["psr", "lora"])
@pytest.mark.parametrize("chunk_tokens", [1, 3, 5, 10_000])
def test_chunked_training_equals_full(tiny_llama, kind, chunk_tokens):
    """One epoch, bs 2 / accum 3 over six unequal rows, both object
    classes (PSR with the gate regulariser live), chunk sizes that split
    rows mid-way (1, 3, 5) and one that makes a single chunk (10k): the
    trained parameters agree to ~fp32 precision, so the accumulated
    gradient — clip + AdamW downstream — was the same."""
    model, tok = tiny_llama
    exs = _examples(tok, 6)
    kw = dict(lr=1e-3, epochs=1, max_len=256, seed=0, batch_size=2,
              accum_steps=3, lr_schedule="constant", max_grad_norm=math.inf,
              reg_coeff=(0.1 if kind == "psr" else 0.0))

    def run(impl):
        obj = _fresh(kind, model)
        diag: dict = {}
        hist = train_object_ce(model, tok, obj, exs, loss_impl=impl,
                               chunk_tokens=chunk_tokens, diagnostics=diag,
                               **kw)
        return _params(obj), hist, diag

    p_full, h_full, d_full = run("full")
    p_chunk, h_chunk, d_chunk = run("chunked_ce")
    assert len(h_full) == len(h_chunk) == 1
    assert abs(h_full[0] - h_chunk[0]) <= 1e-5 * max(1.0, abs(h_full[0]))
    moved = 0.0
    for a, b in zip(p_full, p_chunk):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), (
            f"{kind}/chunk={chunk_tokens}: chunked != full "
            f"(max abs diff {float((a - b).abs().max()):.3e})")
        moved += float(a.abs().sum())
    assert moved > 0.0
    assert d_full["loss_impl"] == "full" and d_full["chunk_tokens"] is None
    assert d_chunk["loss_impl"] == "chunked_ce"
    assert d_chunk["chunk_tokens"] == chunk_tokens
    # the startup cross-check ran (on its bounded slice) and agreed
    assert d_chunk["chunk_check"]["rel_diff"] <= 1e-6
    assert d_chunk["chunk_check"]["check_positions"] <= 1024
    assert d_chunk["chunk_check"]["check_targets"] >= 1
    assert d_full["chunk_check"] is None
    # token accounting identical between the two
    for k in ("tokens_loss_total", "tokens_real_total", "tokens_padded_total"):
        assert d_full[k] == d_chunk[k] > 0


def _micro_grads(model, tok, obj, exs, *, impl, reg_coeff, chunk_tokens=3):
    """Gradient of ONE micro-batch's step_loss (w_ce = 1) w.r.t. the
    object's params, by the same assembly the trainer uses — an
    independent direct comparison that does not go through AdamW."""
    enc = [_encode_aligned(tok, ex, 256)[1] for ex in exs]
    batch = collate(enc, tok.pad_token_id)
    n_tok = int((batch.labels[:, 1:] != -100).sum())
    for p in obj.trainable:
        p.grad = None
    with obj.apply(model, scale=1.0):
        if impl == "full":
            logits = model(input_ids=batch.input_ids,
                           attention_mask=batch.attention_mask).logits
            s_logits = _gather_completion_logits(logits, batch)
            labels = batch.labels[:, 1:]
            targets = labels[labels != -100]
            loss = F.cross_entropy(s_logits.float(), targets)
            if reg_coeff:
                loss = loss + reg_coeff * obj.gate_reg_term(
                    batch.attention_mask, None)
            loss.backward()
            ce = float(loss)
        else:
            h = cc.forward_last_hidden(model, batch.input_ids,
                                       batch.attention_mask)
            h_sel, targets = cc.completion_hidden(h, batch)
            nll, g = cc.chunked_ce(cc.lm_head_of(model),
                                   cc.final_logit_softcap(model), h_sel,
                                   targets, chunk_tokens=chunk_tokens,
                                   grad_scale=1.0 / n_tok)
            ce = float(nll / n_tok)
            if reg_coeff:
                reg = reg_coeff * obj.gate_reg_term(batch.attention_mask, None)
                torch.autograd.backward([h_sel, reg], [g, None])
                ce += float(reg)
            else:
                torch.autograd.backward([h_sel], [g])
    grads = [p.grad.detach().clone() for p in obj.trainable]
    return ce, grads


@pytest.mark.parametrize("kind,reg", [("psr", 0.1), ("psr", 0.0),
                                      ("lora", 0.0)])
def test_direct_gradient_parity(tiny_llama, kind, reg):
    """Direct: the assembled dL/dparams of one ragged micro-batch (a row
    whose prompt masks most positions included) match between the two
    paths to <= 1e-5 relative, and the losses agree."""
    model, tok = tiny_llama
    exs = _examples(tok, 6)
    obj = _fresh(kind, model)
    # nudge the object off its zero init so gradients are non-trivial
    torch.manual_seed(0)
    with torch.no_grad():
        for p in obj.trainable:
            p.add_(torch.randn_like(p) * 1e-2)
    ce_f, g_f = _micro_grads(model, tok, obj, exs, impl="full",
                             reg_coeff=reg)
    ce_c, g_c = _micro_grads(model, tok, obj, exs, impl="chunked_ce",
                             reg_coeff=reg)
    assert abs(ce_f - ce_c) <= 1e-6 * max(1.0, abs(ce_f))
    num = math.sqrt(sum(float(((a - b) ** 2).sum()) for a, b in zip(g_f, g_c)))
    den = math.sqrt(sum(float((a ** 2).sum()) for a in g_f))
    assert den > 0.0
    assert num / den <= 1e-5, f"relative grad diff {num / den:.3e}"


def test_parity_helper_is_exact_in_fp32_and_restores_the_object(tiny_llama):
    from sisteer.extraction.objects import ce_loss_impl_parity

    model, tok = tiny_llama
    exs = _examples(tok, 6)
    obj = _fresh("psr", model)
    before = _params(obj)
    rep = ce_loss_impl_parity(model, tok, obj, exs[2:4], max_len=256,
                              reg_coeff=0.1, chunk_tokens=3, perturb=1e-3)
    assert rep["rows"] == 2 and rep["n_targets"] > 0
    assert rep["ce_rel_diff"] <= 1e-6
    # (the cosine is itself fp32-accumulated, so ~1e-7 is its own floor)
    assert rep["grad_rel_l2_diff"] <= 1e-5 and rep["grad_cosine"] > 1 - 1e-6
    # fp32 CPU, deterministic kernels: every comparison leg is (near) zero
    assert rep["repeat_full_grad_rel_l2_diff"] == 0.0
    for k in ("head_paired", "chunk_split", "single_chunk_vs_full",
              "rowwise_sensitivity"):
        assert rep[f"{k}_grad_rel_l2_diff"] <= 1e-5, k
        assert rep[f"{k}_grad_cosine"] > 1 - 1e-6, k
    assert rep["head_paired_ce_rel_diff"] <= 1e-6
    assert rep["head_paired_grad_l2_full"] > 0.0
    # the note carries NO bound semantics
    note = rep["calibration_note"]
    assert "no bound semantics" in note and "upper" not in note.lower()
    assert "not a proof" not in note or "None of these is a proof" in note
    # the object's parameters and grads are restored after the probe
    for a, b in zip(before, _params(obj)):
        assert torch.equal(a, b)
    assert all(p.grad is None for p in obj.trainable)


def test_chunked_ce_edges():
    """The helper on its own: N==0 returns (0, empty grad) without an
    autograd complaint; a chunk size not dividing N covers every row
    exactly once; grad equals the dense autograd gradient."""
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 13, bias=False)
    head.weight.requires_grad_(False)
    h = torch.randn(7, 8, requires_grad=True)
    t = torch.randint(0, 13, (7,))
    nll, g = cc.chunked_ce(head, None, h, t, chunk_tokens=3, grad_scale=0.5)
    ref = F.cross_entropy(head(h).float(), t, reduction="sum")
    (g_ref,) = torch.autograd.grad(0.5 * ref, h)
    assert torch.allclose(nll, ref.detach(), rtol=1e-6, atol=1e-6)
    assert torch.allclose(g, g_ref, rtol=1e-6, atol=1e-7)
    # no-grad evaluation
    nll2, g2 = cc.chunked_ce(head, None, h.detach(), t, chunk_tokens=2)
    assert g2 is None and torch.allclose(nll2, ref.detach())
    # the empty edge (a whole batch with no targets; the trainer refuses
    # it upstream, the helper must still be well-defined)
    nll0, g0 = cc.chunked_ce(head, None, h[:0].detach(), t[:0],
                             chunk_tokens=3, grad_scale=1.0)
    assert float(nll0) == 0.0 and g0.shape == (0, 8)
    with pytest.raises(ValueError, match="targets"):
        cc.chunked_ce(head, None, h, t[:5], chunk_tokens=3)
    with pytest.raises(ValueError, match="chunk_tokens"):
        cc.chunked_ce(head, None, h, t, chunk_tokens=0)


def test_frozen_head_and_body_resolution(tiny_llama):
    model, _tok = tiny_llama
    assert cc.body_module(model) is model.model
    assert cc.final_logit_softcap(model) is None
    for p in model.parameters():
        p.requires_grad_(False)
    assert cc.lm_head_of(model) is model.lm_head
    model.lm_head.weight.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="trainable"):
            cc.lm_head_of(model)
    finally:
        model.lm_head.weight.requires_grad_(False)


def test_forward_last_hidden_refuses_a_wrapper_that_ignores_logits_to_keep():
    """A wrapper that materialises full logits despite logits_to_keep=1
    would silently defeat the point — the shape assertion catches it."""
    class Body(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kw):
            b, s = input_ids.shape
            return torch.zeros(b, s, 4)

    class Wrapper(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self):
            super().__init__()
            self.model = Body()
            self.lm_head = torch.nn.Linear(4, 9, bias=False)

        def forward(self, input_ids=None, attention_mask=None,
                    logits_to_keep=0):
            h = self.model(input_ids=input_ids, attention_mask=attention_mask)
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.lm_head(h))  # ignores the arg

    ids = torch.zeros(2, 5, dtype=torch.long)
    with pytest.raises(RuntimeError, match="logits_to_keep"):
        cc.forward_last_hidden(Wrapper(), ids, torch.ones_like(ids))


# -- (b) the final logit softcap is replicated ---------------------------------


@pytest.fixture(scope="module")
def tiny_gemma2():
    """A from-config tiny Gemma2ForCausalLM (random init, fp32) — the
    architecture family whose causal-LM forward applies
    final_logit_softcapping AFTER lm_head, exactly as gemma-4's wrapper
    does (modeling_gemma4: divide, tanh, multiply)."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    cfg = Gemma2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=8,
                       max_position_embeddings=64,
                       final_logit_softcapping=30.0,
                       attn_logit_softcapping=None, sliding_window=16,
                       pad_token_id=0, bos_token_id=1, eos_token_id=2,
                       attn_implementation="eager")
    model = Gemma2ForCausalLM(cfg).eval()
    # random weights are small; scale the head so logits reach the
    # softcap's nonlinear range and the cap actually changes the loss
    with torch.no_grad():
        model.lm_head.weight.mul_(40.0)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_softcap_replicated_when_present(tiny_gemma2):
    model = tiny_gemma2
    assert cc.final_logit_softcap(model) == 30.0
    torch.manual_seed(1)
    ids = torch.randint(3, 64, (2, 11))
    attn = torch.ones_like(ids)
    attn[1, 8:] = 0
    labels = ids.clone()
    labels[:, :3] = -100
    labels[1, 8:] = -100
    from sisteer.extraction.traindata import Batch
    batch = Batch(ids, attn, torch.tensor([3, 3]), labels)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attn).logits
        ref = F.cross_entropy(
            _gather_completion_logits(logits, batch).float(),
            labels[:, 1:][labels[:, 1:] != -100], reduction="sum")
        h = cc.forward_last_hidden(model, ids, attn)
        h_sel, targets = cc.completion_hidden(h, batch)
        with_cap, _ = cc.chunked_ce(model.lm_head, 30.0, h_sel, targets,
                                    chunk_tokens=4)
        no_cap, _ = cc.chunked_ce(model.lm_head, None, h_sel, targets,
                                  chunk_tokens=4)
    assert torch.allclose(with_cap, ref.detach(), rtol=1e-6, atol=1e-5)
    # the cap is load-bearing on this model: dropping it moves the loss,
    # so a chunked path that forgot it could not pass the line above
    assert abs(float(no_cap) - float(ref)) > 1e-2 * max(1.0, float(ref))


def test_softcap_trainer_end_to_end_on_gemma2(tiny_gemma2):
    """Through the trainer itself on the softcapped model: chunked ==
    full including the softcap (a tiny tokenizer-free run — examples are
    encoded through the Llama tokenizer's ids only for their shapes, so
    build the batch straight from ids via a stub tokenizer)."""
    model = tiny_gemma2

    class Tok:
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=True):
            ids = [(ord(c) % 60) + 3 for c in text]
            if add_special_tokens:
                ids = [1] + ids
            return {"input_ids": ids}

    tok = Tok()
    from sisteer.extraction import tokcache

    # The token cache is process-global and keyed on text only: install a
    # private one for the stub tokenizer and RESTORE the previous cache
    # after (test_doc_mode's precedent), or every later tokenizer in the
    # session would read this stub's ids.
    prev = tokcache.active()
    tokcache.set_active(tokcache.TokCache())
    try:
        exs = _examples(tok, 4)
        kw = dict(lr=1e-3, epochs=1, max_len=128, seed=0, batch_size=2,
                  accum_steps=2, lr_schedule="constant",
                  max_grad_norm=math.inf)

        def run(impl):
            obj = PSRObject(model, [0, 1], train_direction=True)
            train_object_ce(model, tok, obj, exs, loss_impl=impl,
                            chunk_tokens=5, reg_coeff=0.1, **kw)
            return _params(obj)

        for a, b in zip(run("full"), run("chunked_ce")):
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)
    finally:
        tokcache.set_active(prev)


# -- (c) the fingerprint moves on loss_impl, and on nothing else ---------------


def test_fingerprint_differs_by_loss_impl_only(tmp_path):
    import train_two_si_objects as t2

    def cfg(**arm_over):
        return {
            "model": "m", "model_revision": "r", "layer": 1, "max_len": 64,
            "data_mode": "documents", "si_ids": [SI],
            "batch_size_by_arm": {"psr_ce_all": 1},
            "train_kwargs": {
                "default": {"holdout_frac": 0.0, "positions": "all"},
                "psr_ce_all": {"lr": 1e-3, "lr_schedule": "warmup_cosine",
                               "warmup_frac": 0.05, "max_grad_norm": 0.1,
                               "weight_decay": 0.01, "gate_reg": 0.1,
                               "accum_steps": 8, **arm_over},
            },
        }

    common = dict(positions="all", pool_sha256="pool", config_sha="cfg")
    base = t2.cell_fingerprint_fields(cfg(), SI, "psr_ce_all", **common)
    chunk = t2.cell_fingerprint_fields(cfg(loss_impl="chunked_ce"), SI,
                                       "psr_ce_all", **common)
    explicit_full = t2.cell_fingerprint_fields(cfg(loss_impl="full"), SI,
                                               "psr_ce_all", **common)
    # every field equal except the optimizer's loss_impl slot
    for k in base:
        if k != "optimizer":
            assert base[k] == chunk[k]
    diff = {k for k in set(base["optimizer"]) | set(chunk["optimizer"])
            if base["optimizer"].get(k) != chunk["optimizer"].get(k)}
    assert diff == {"loss_impl"}
    assert "loss_impl" not in base["optimizer"]
    assert chunk["optimizer"]["loss_impl"] == "chunked_ce"
    # an explicit "full" is the same work as the historical default
    assert explicit_full == base
    fp = t2.cell_fit_fingerprint
    assert fp(cfg(loss_impl="chunked_ce"), SI, "psr_ce_all", **common) \
        != fp(cfg(), SI, "psr_ce_all", **common) \
        == fp(cfg(loss_impl="full"), SI, "psr_ce_all", **common)
    assert t2.resolved_loss_impl(cfg(), "psr_ce_all") == "full"
    assert t2.resolved_loss_impl(cfg(loss_impl="chunked_ce"),
                                 "psr_ce_all") == "chunked_ce"


def test_config_guard_refuses_bad_or_default_level_loss_impl():
    import train_two_si_objects as t2

    def cfg(default_extra=None, arm_extra=None):
        return {
            "data_mode": "documents",
            "train_kwargs": {
                "default": {"holdout_frac": 0.0, "positions": "all",
                            **(default_extra or {})},
                "psr_ce_all": {"floor_ratio_max": None, **(arm_extra or {})},
            },
        }

    t2.refuse_ce_misconfiguration(cfg(), ["psr_ce_all"])
    t2.refuse_ce_misconfiguration(cfg(arm_extra={"loss_impl": "chunked_ce"}),
                                  ["psr_ce_all"])
    with pytest.raises(SystemExit, match="loss_impl"):
        t2.refuse_ce_misconfiguration(cfg(arm_extra={"loss_impl": "fused"}),
                                      ["psr_ce_all"])
    with pytest.raises(SystemExit, match="loss_impl"):
        t2.refuse_ce_misconfiguration(
            cfg(default_extra={"loss_impl": "chunked_ce"}), ["psr_ce_all"])


# -- max_steps: a bounded run --------------------------------------------------


def test_max_steps_bounds_the_run_and_refuses_publishable_combos(tiny_llama):
    model, tok = tiny_llama
    exs = _examples(tok, 6)
    diag: dict = {}
    hist = train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-3,
                           epochs=5, max_len=256, batch_size=1, accum_steps=2,
                           max_steps=2, diagnostics=diag,
                           loss_impl="chunked_ce")
    assert len(hist) == 2 and diag["optimizer_steps_run"] == 2
    assert diag["stopped_by_max_steps"] is True
    assert diag["epochs_run"] == 0 and diag["heldout_curve"] == []
    assert diag["train_wall_s"] > 0.0
    with pytest.raises(ValueError, match="max_steps"):
        train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-3,
                        epochs=1, max_len=256, batch_size=2, max_steps=1,
                        heldout=exs[:2],
                        early_stop=__import__(
                            "sisteer.extraction.distill",
                            fromlist=["EarlyStopSpec"]).EarlyStopSpec())
    with pytest.raises(ValueError, match="loss_impl"):
        train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-3,
                        epochs=1, max_len=256, batch_size=2,
                        loss_impl="liger")


def test_cross_check_slice_is_bounded_never_extended(tiny_llama):
    """The startup cross-check scores ONE slice of <= chunk_check_max_len
    positions, taken from the FIRST (shortest) row that has a CE target
    inside the cap — a row whose masked prefix exceeds the cap is SKIPPED,
    never extended (an extension could materialise near-full logits
    before the loop), and a pool with no qualifying row is
    refused. Never past a row's real length (padding drops with the tail)."""
    from sisteer.extraction.objects import _chunk_check_slice

    model, tok = tiny_llama
    long_prefix = "Sys " * 12 + "\n\n"          # ~13-token masked prefix
    exs = docs_to_ce_examples(tok, [
        # the SHORTEST row (~17 tokens) has the long prefix + a tiny target
        Pair(prompt=long_prefix, pos="T.", neg="", instruction_id=SI),
        # ... the next-shortest has an empty prompt and a target at once
        Pair(prompt="", pos="A plain story body with rather more words in "
                            "it than the tiny target row carries, so it "
                            "sorts after it. " * 3, neg="", instruction_id=SI),
    ])
    enc = sorted((_encode_aligned(tok, ex, 256)[1] for ex in exs),
                 key=lambda e: len(e.ids))
    # premise: the long-prefix row (~31 tokens on this tokenizer) IS first
    assert int(enc[0].prompt_len) > 8 and len(enc[0].ids) < len(enc[1].ids)
    batches = [collate(enc[i:i + 1], tok.pad_token_id) for i in range(2)]
    # cap 8 (< the first row's prompt): the first row is skipped, the second
    # is cut to exactly the cap, and the cut carries a target
    sl = _chunk_check_slice(batches, 8)
    assert sl.input_ids.shape == (1, 8)
    assert int(sl.prompt_lens[0]) == enc[1].prompt_len
    assert int((sl.labels[:, 1:] != -100).sum()) >= 1
    # a cap above every row's length: the FIRST row qualifies and is
    # returned whole — no padding, nothing past its real length
    both = [collate(enc, tok.pad_token_id)]
    sl = _chunk_check_slice(both, 10_000)
    n0 = len(enc[0].ids)
    assert sl.input_ids.shape[1] == n0 and int(sl.attention_mask.sum()) == n0
    # only long-prefix rows under a small cap: refused, not extended
    with pytest.raises(ValueError, match="no row has a CE target"):
        _chunk_check_slice(batches[:1], 8)
    # through the trainer: bounded, with a target, and exact
    diag: dict = {}
    train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-3,
                    epochs=1, max_len=256, batch_size=1, accum_steps=2,
                    loss_impl="chunked_ce", chunk_check_max_len=8,
                    diagnostics=diag)
    cc = diag["chunk_check"]
    assert cc["check_positions"] <= 8 and cc["check_targets"] >= 1
    assert cc["rel_diff"] <= 1e-6
    # the trainer refuses a pool with no qualifying row (chunked only)
    with pytest.raises(ValueError, match="no row has a CE target"):
        train_object_ce(model, tok, _fresh("lora", model), exs[:1], lr=1e-3,
                        epochs=1, max_len=256, batch_size=1,
                        loss_impl="chunked_ce", chunk_check_max_len=8)
    train_object_ce(model, tok, _fresh("lora", model), exs[:1], lr=1e-3,
                    epochs=1, max_len=256, batch_size=1, loss_impl="full")
    with pytest.raises(ValueError, match="chunk_check_max_len"):
        train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-3,
                        epochs=1, max_len=256, batch_size=2,
                        loss_impl="chunked_ce", chunk_check_max_len=1)


def test_per_step_throughput_record(tiny_llama):
    """Every optimizer-step record carries the token counts and the wall
    seconds the heartbeat / bench read; the loss-token count equals the
    trainer's own denominator."""
    model, tok = tiny_llama
    exs = _examples(tok, 4)
    recs: list = []
    train_object_ce(model, tok, _fresh("lora", model), exs, lr=1e-3,
                    epochs=1, max_len=256, batch_size=2, accum_steps=1,
                    log_fn=lambda d, step=None: recs.append((step, d)),
                    log_tag="t")
    steps = [d for s, d in recs if "t/ce_loss" in d]
    assert len(steps) == 2
    enc = [_encode_aligned(tok, ex, 256)[1] for ex in exs]
    enc = sorted(enc, key=lambda e: len(e.ids))
    exp_loss = [int((collate(enc[i:i + 2], tok.pad_token_id).labels[:, 1:]
                     != -100).sum()) for i in (0, 2)]
    got = sorted(int(d["t/loss_tokens"]) for d in steps)
    assert got == sorted(exp_loss)
    for d in steps:
        assert d["t/real_tokens"] <= d["t/padded_tokens"]
        assert d["t/sec_per_step"] > 0.0


# -- the plumbing around the trainer -------------------------------------------


def test_memprobe_variant_suffix_never_touches_the_folded_name():
    import importlib.util
    import types

    # sdfbase_fits imports modal at module level; a duck-typed stub is the
    # test_sdfbase_fits_app convention.
    fake = types.ModuleType("modal")

    class _App:
        def __init__(self, name):
            self.name = name

        def function(self, *a, **k):
            return lambda fn: fn

        def local_entrypoint(self, *a, **k):
            return lambda fn: fn

    class _Chain:
        def __getattr__(self, name):
            return lambda *a, **k: self

    fake.App = _App
    fake.Image = _Chain()
    fake.Volume = types.SimpleNamespace(from_name=lambda *a, **k: object())
    fake.Secret = types.SimpleNamespace(from_name=lambda *a, **k: object())
    fake.Function = types.SimpleNamespace(from_name=lambda *a, **k: None)
    saved = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        spec = importlib.util.spec_from_file_location(
            "_sdfbase_fits_for_suffix",
            Path(__file__).resolve().parents[1]
            / "scripts/modal/sdfbase_fits.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["modal"] = saved
        else:
            sys.modules.pop("modal", None)
    f = mod.memprobe_variant_suffix
    # EMPTY only when neither option was supplied; ANY explicit value —
    # a baseline re-probe with --loss-impl full included — gets its own
    # name, so the folded artifact is never overwritten.
    assert f() == "" and f("", "") == ""
    assert f("full") == "__full" and f("config") == "__config"
    assert f("chunked_ce") == "__chunked_ce"
    assert f("chunked_ce", "on") == "__chunked_ce__gcon"
    assert f("", "off") == "__gcoff" and f("full", "config") == "__full__gcconfig"
    # the bench falls back to a REGISTERED label for an unregistered pool
    # cell and records both
    assert mod.BENCH_SI_LABEL_FALLBACK == "constv2-mix-full"


def test_heartbeat_prints_tok_per_s_when_fed_tokens(capsys):
    from sisteer.extraction.fitmon import GpuDutySampler, Heartbeat

    hb = Heartbeat(total_steps=10, interval=3600.0, gpu_duty=False)
    try:
        hb.step(1, 2.0, 1e-3, tokens=1000)
        hb.step(2, 1.9, 1e-3, tokens=1000)
        hb._line()
        out = capsys.readouterr().out
        assert "tok/s=" in out and "step 2/10" in out
        assert "sm=" not in out  # sampler disabled -> no duty column
        assert hb.duty_summary() is None
        # a trainer without a token record prints no tok/s column
        hb2 = Heartbeat(total_steps=3, interval=3600.0, gpu_duty=False)
        hb2.step(1, 1.0, None)
        hb2._line()
        assert "tok/s=" not in capsys.readouterr().out
        hb2.close()
        s = GpuDutySampler(enabled=False)
        assert s.window(0.0) is None and s.summary() is None
        s.close()
    finally:
        hb.close()


def test_tune_arm_ce_variant_knobs_refused_for_non_ce_arms():
    import tune_arm
    from types import SimpleNamespace

    ad = tune_arm.make_adapter("lora_r1_all")
    with pytest.raises(SystemExit, match="CE-arm knobs"):
        tune_arm._apply_ce_variant(ad, SimpleNamespace(loss_impl="chunked_ce",
                                                       chunk_tokens=0))
    tune_arm._apply_ce_variant(ad, SimpleNamespace(loss_impl="config",
                                                   chunk_tokens=0))
    ce = tune_arm.make_adapter("psr_ce_all")
    tune_arm._apply_ce_variant(ce, SimpleNamespace(loss_impl="chunked_ce",
                                                   chunk_tokens=512))
    assert ce.loss_impl_override == "chunked_ce"
    assert ce.chunk_tokens_override == 512
    assert ce.accum_for(2) == 4
    ce.accum_override = 3
    assert ce.accum_for(2) == 3
