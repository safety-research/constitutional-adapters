"""Chunked LM-head cross-entropy — the ``loss_impl: "chunked_ce"`` path of
:func:`~sisteer.extraction.objects.train_object_ce`: the same document-CE
objective computed WITHOUT ever materialising full-sequence logits.

Why: at a 262k-token vocabulary (gemma-4) and ``max_len`` 8192 the
``[B, S, V]`` logits of the ``"full"`` path cost ~4.3 GB in bf16 per row,
~8.6 GB once ``.float()``-ed for the loss and again for ``log_softmax``'s
saved output — ~28 GB per SEQUENCE that has nothing to do with the model
body, and the reason the gemma cells memprobed to micro-batch 1 (85.5 GiB
at bs 1, OOM at 2). The body's activations at bs 1 are ~41 GB; the logits
were the other half.

How (the standard chunked-loss decomposition, Liger-style but dependency
free and generic over any ``lm_head`` + optional final logit softcap):

1. ONE body forward through the model's own ``forward`` — never the inner
   module directly, because steering objects register pre-hooks on the
   TOP-LEVEL model (:class:`~sisteer.extraction.psr.PSRObject`
   ``_track_model_inputs``, :class:`~sisteer.hooks.SteeringHandle`) and a
   body-only call would silently skip them. ``logits_to_keep=1`` makes the
   wrapper's own ``lm_head`` call a ``[B, 1, V]`` no-op, and a forward
   hook on the body module captures ``last_hidden_state`` (``h``), the
   tensor the wrapper would have fed to ``lm_head``.
2. Gather FIRST: ``h[:, :-1][mask]`` picks exactly the hidden states at
   completion-predicting positions — the same ``logits[:, :-1][mask]``
   gather :func:`~sisteer.extraction.distill._gather_completion_logits`
   applies to logits. ``lm_head`` is position-wise, so gathering before it
   is exact and skips every prompt/padding position outright.
3. Loop over chunks of ``chunk_tokens`` target rows: ``lm_head(h_c)`` in
   the model's dtype, the model's own final logit softcap when its config
   declares one (``final_logit_softcapping`` — gemma-2/3/4; replicated op
   for op: ``/ cap``, ``tanh``, ``* cap``), then ``F.cross_entropy(
   logits.float(), targets_c, reduction="sum")`` — the fp32 cast the
   ``"full"`` path performs on its gathered logits, now per chunk. Under
   grad, ``torch.autograd.grad`` of the SCALED chunk loss w.r.t. the
   DETACHED chunk input yields that chunk's ``dL/dh_c`` immediately and
   frees its logits; the pieces are written into one ``dL/dh_sel`` buffer.
4. ONE body backward: ``h_sel.backward(dL/dh_sel)`` (jointly with any
   auxiliary term such as the PSR gate regulariser, in a single
   ``torch.autograd.backward`` call) — the chain rule makes this EXACTLY
   the gradient the full path would have produced; the only differences
   are bf16 kernel-tiling noise in ``lm_head`` (M = chunk rows instead of
   B*S rows) and fp32 summation order across chunks.

Peak memory = body activations + ONE chunk's logits (bf16 + fp32 + the
softmax's fp32 saved output; 2048 rows x 262k x 10 B ~ 5.4 GB) instead of
the whole sequence's.

The LM head must be FROZEN (no trainable parameter), asserted at resolve
time: step 3 differentiates w.r.t. the chunk input only, and a trainable
head would silently receive no gradient. Every CE arm of this campaign
(PSR gates/directions on decoder layers, o_proj LoRA) leaves it frozen;
the tied input embedding is frozen with it.
"""
from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn

#: The trainer's loss implementations. ``"full"`` is the historical path
#: (full-sequence logits; every published fit); ``"chunked_ce"`` is this
#: module. A different implementation is a DIFFERENT fit identity (bf16
#: numerics differ at the kernel level) and enters the fit fingerprint.
LOSS_IMPLS = ("full", "chunked_ce")

#: Default rows per chunk. Sized for the worst vocabulary in the campaign
#: (262,144): 2048 x 262,144 x (2 B bf16 logits + 4 B fp32 cast + 4 B saved
#: log-softmax) ~ 5.4 GB transient — small next to the body's activations,
#: large enough that the per-chunk launch overhead is amortised.
DEFAULT_CHUNK_TOKENS = 2048


def final_logit_softcap(model) -> float | None:
    """The model's ``final_logit_softcapping`` (gemma-2/3/4's tanh cap on
    the LM-head output, applied INSIDE the causal-LM forward after
    ``lm_head``), or ``None`` when the architecture has none. Read from the
    TEXT config (``config.get_text_config()`` — the multimodal wrappers
    nest it), the same attribute the wrapper's own forward consults."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    cap = getattr(text, "final_logit_softcapping", None)
    if cap is None:
        cap = getattr(cfg, "final_logit_softcapping", None)
    return float(cap) if cap is not None else None


def body_module(model) -> nn.Module:
    """The module whose output the causal-LM wrapper feeds to ``lm_head``:
    ``model.<base_model_prefix>`` (``model.model`` for Llama / Qwen3.5 /
    Gemma-4 ``ForCausalLM`` and ``ForConditionalGeneration`` alike — the
    wrapper does ``outputs = self.model(...)``; ``hidden_states =
    outputs.last_hidden_state``; ``self.lm_head(hidden_states[...])``).
    Refuses rather than guesses when the prefix is missing or is not a
    module."""
    prefix = getattr(model, "base_model_prefix", None) or "model"
    body = getattr(model, prefix, None)
    if not isinstance(body, nn.Module) or body is model:
        raise TypeError(
            f"chunked_ce: cannot resolve the body module of "
            f"{type(model).__name__} (base_model_prefix={prefix!r}) — the "
            "chunked path captures the wrapper's last_hidden_state from "
            "that module; refusing to guess.")
    return body


def lm_head_of(model) -> nn.Module:
    """``model.lm_head``, asserted FROZEN — the chunked path differentiates
    each chunk w.r.t. its hidden-state input only, so a trainable head
    would silently receive no gradient."""
    head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Module):
        raise TypeError(
            f"chunked_ce: {type(model).__name__} exposes no lm_head module; "
            "the chunked path needs the wrapper's LM head to apply per chunk.")
    trainable = [n for n, p in head.named_parameters() if p.requires_grad]
    if trainable:
        raise ValueError(
            f"chunked_ce: lm_head has trainable parameter(s) {trainable} — "
            "the chunked path assumes a frozen LM head (its per-chunk "
            "autograd.grad targets the hidden-state input only), so those "
            "parameters would silently get no gradient. Use loss_impl "
            "'full' for an object that trains the head.")
    return head


def _accepts_logits_to_keep(model) -> bool:
    try:
        return "logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def forward_last_hidden(model, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
    """One forward through ``model``'s OWN ``forward`` (so every hook a
    steering object registered — on decoder layers AND on the top-level
    model — fires exactly as in the full path), returning the body's
    ``last_hidden_state`` ``[B, S, d]`` with its autograd graph intact.

    ``logits_to_keep=1`` reduces the wrapper's ``lm_head`` call to one
    position per row; the returned logits' shape is ASSERTED ``[B, 1, V]``
    so a wrapper that ignored the argument (and materialised the full
    logits behind our back) fails loudly instead of silently defeating the
    point. The hidden state is captured by a forward hook on the body
    module (:func:`body_module`) registered for this call only."""
    if not _accepts_logits_to_keep(model):
        raise TypeError(
            f"chunked_ce: {type(model).__name__}.forward takes no "
            "logits_to_keep — the chunked path cannot suppress the wrapper's "
            "full-sequence lm_head call on this architecture; use loss_impl "
            "'full'.")
    body = body_module(model)
    captured: list[torch.Tensor] = []

    def _capture(_mod, _args, out):
        h = getattr(out, "last_hidden_state", None)
        if h is None:
            h = out[0] if isinstance(out, (tuple, list)) else out
        captured.append(h)

    handle = body.register_forward_hook(_capture)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    logits_to_keep=1)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"chunked_ce: expected exactly one body forward, captured "
            f"{len(captured)} — the wrapper's forward does not call the "
            "body module once as assumed.")
    logits = getattr(out, "logits", None)
    if logits is None or logits.dim() != 3 or logits.shape[1] != 1:
        raise RuntimeError(
            "chunked_ce: the wrapper did not honour logits_to_keep=1 "
            f"(logits shape {None if logits is None else tuple(logits.shape)})"
            " — it materialised more than one position's logits; refusing "
            "to run a 'chunked' path that still pays for the full logits.")
    h = captured[0]
    if not isinstance(h, torch.Tensor) or h.dim() != 3 \
            or h.shape[:2] != input_ids.shape:
        raise RuntimeError(
            "chunked_ce: captured hidden state has shape "
            f"{None if not isinstance(h, torch.Tensor) else tuple(h.shape)}, "
            f"expected [B={input_ids.shape[0]}, S={input_ids.shape[1]}, d].")
    return h


def chunked_ce(lm_head: nn.Module, softcap: float | None,
               h_sel: torch.Tensor, targets: torch.Tensor, *,
               chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
               grad_scale: float | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Summed token NLL over ``targets`` from gathered hidden states
    ``h_sel`` ``[N, d]``, ``chunk_tokens`` rows at a time.

    Returns ``(nll_sum, grad)``: ``nll_sum`` a detached fp32 scalar (the
    sum over ALL N targets — the caller divides, exactly as the full path's
    ``F.cross_entropy(..., reduction="mean")`` divides by N); ``grad`` is
    ``dL/dh_sel`` ``[N, d]`` (``h_sel``'s dtype) for
    ``L = grad_scale * nll_sum`` when ``grad_scale`` is given, else
    ``None`` (a no-grad evaluation). The caller backpropagates ``grad``
    through the body with ``torch.autograd.backward([h_sel], [grad])``.

    Per chunk the logits are computed in the head's dtype, softcapped when
    ``softcap`` is set (op for op as the wrapper does it), cast to fp32 for
    the loss, and — under grad — differentiated w.r.t. the DETACHED chunk
    input at once, so at most one chunk's logits are alive. Chunks never
    span the ``targets`` boundary; ``N == 0`` returns ``(0.0, empty)``.
    Autograd inside is disabled when ``grad_scale`` is None."""
    chunk = int(chunk_tokens)
    if chunk < 1:
        raise ValueError(f"chunk_tokens must be >= 1, got {chunk_tokens!r}")
    n = int(h_sel.shape[0])
    if int(targets.numel()) != n:
        raise ValueError(
            f"chunked_ce: {n} gathered hidden rows vs {int(targets.numel())} "
            "targets — the gather masks disagree.")
    device = h_sel.device
    nll_total = torch.zeros((), dtype=torch.float32, device=device)
    want_grad = grad_scale is not None
    grad = torch.empty_like(h_sel) if want_grad else None
    if n == 0:
        return nll_total, grad
    ctx = torch.enable_grad() if want_grad else torch.no_grad()
    with ctx:
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            hc = h_sel[c0:c1]
            if want_grad:
                hc = hc.detach().requires_grad_(True)
            logits = lm_head(hc)
            if softcap is not None:
                # Op for op the wrapper's own sequence (modeling_gemma*):
                # divide, tanh, multiply — in the logits' dtype.
                logits = logits / softcap
                logits = torch.tanh(logits)
                logits = logits * softcap
            nll = F.cross_entropy(logits.float(), targets[c0:c1],
                                  reduction="sum")
            if want_grad:
                (g,) = torch.autograd.grad(nll * float(grad_scale), hc)
                grad[c0:c1] = g
            nll_total = nll_total + nll.detach()
    return nll_total, grad


def completion_hidden(h: torch.Tensor, batch) -> tuple[torch.Tensor,
                                                       torch.Tensor]:
    """``(h_sel, targets)``: hidden states at completion-PREDICTING
    positions and the completion tokens they predict — the hidden-state
    twin of ``_gather_completion_logits`` + ``labels[labels != -100]``
    (same mask, same row-major flatten order), so the two paths score
    exactly the same tokens in the same order."""
    labels = batch.labels[:, 1:]
    mask = labels != -100
    return h[:, :-1][mask], labels[mask]
