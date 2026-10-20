"""Object-axis ablation: the SAME full-distribution KL objective as
distill_soft, compiled into different intervention OBJECTS.

The loss axis (soft vs hard vs preference) is measured by
distill_soft / distill_hard / bidpo. This module holds the other axis —
what the trained thing IS — with the loss, pairs, data, and model fixed:

- vector (baseline cell): additive residual-stream vector, one layer,
  all positions — trained by distill.train_distill (d params).
- soft token: ONE trainable input-embedding vector occupying a reserved
  sentinel token inserted right after BOS in the student prompt — the
  Steering Tokens (2601.05062) / Wingate (2210.03162) object (d params).
  At eval, scale multiplies the embedding (scale 1 = as trained).
- rank-1 LoRA: dW = B A^T on ONE weight matrix (attention o_proj) at the
  same layer, B zero-initialized so training starts at the unsteered
  model — the minimal weight-space object, Prompt Baking's (2409.13697)
  direction at matched-order parameter count (2d params). Its residual
  contribution is B * (A . x): the same fixed direction B, but with an
  input-dependent magnitude — the sharpest possible contrast with the
  input-independent vector.

Each object exposes `apply(model, scale)` — a context manager active for
BOTH training forwards and eval (generation, MMLU, ppl) — so every
downstream metric runs identically across objects.
"""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..hooks import PositionMaskedFit, SteeringHandle
from .chunked_ce import (DEFAULT_CHUNK_TOKENS, LOSS_IMPLS, chunked_ce,
                         completion_hidden, final_logit_softcap,
                         forward_last_hidden, lm_head_of)
from .distill import (DistillExample, EarlyStopper, EarlyStopSpec,
                      TokenizeProgress, _encode_aligned,
                      _gather_completion_logits, loss_gather_masks,
                      startup_loss_check, steer_prompt_boundaries)
from .fitckpt import (cuda_rng_states, load_checkpoint,
                      resolve_checkpoint_kwargs, restore_cuda_rng,
                      save_checkpoint, to_cpu)
from ..models import attention_oproj, get_decoder_layers, hidden_size
from .prefix_cache import prefix_cache_for_batches
from .tokcache import cached_ids
from .traindata import Batch, collate

_SENTINEL_PREFS = ("<|reserved_special_token_0|>", "<unused0>", "<|extra_0|>",
                   # gemma-4 / Qwen3.8: <unusedN>/<|extra_0|> are not ADDED
                   # tokens on these tokenizers so they do not encode
                   # atomically; these do (gemma <mask> id 4, Qwen <|fim_pad|>
                   # id 248063). Appended last so Llama-3.1 resolution is
                   # unchanged. Avoid modality tokens (<|image_pad|> etc.):
                   # the ConditionalGeneration forward special-cases them.
                   "<mask>", "<|fim_pad|>",
                   # GLM-4.6V: [sMASK] (id 151332) is an added control token
                   # the chat template never emits; [gMASK] opens every chat
                   # render and [MASK] mirrors BERT-style mask semantics, so
                   # prefer the span-mask sibling that nothing else uses.
                   "[sMASK]",
                   # NemotronH / Nemotron-3-Nano-4B: <SPECIAL_18> ..
                   # <SPECIAL_999> are ~980 reserved single-token control
                   # specials whose id IS the number (<SPECIAL_999> -> 999,
                   # special=True; measured on the pinned tokenizer
                   # 2026-09-02). The chat template emits none of them; the
                   # highest is the one least likely to be claimed by a
                   # future template revision. No earlier entry encodes
                   # atomically on this tokenizer, so appending changes
                   # nothing for the models above.
                   "<SPECIAL_999>")


def find_sentinel(tok) -> tuple[str, int]:
    """A vocab token that never occurs in natural text, to carry the soft
    embedding. Llama-3.1: <|reserved_special_token_0|>; gemma-4: <mask>;
    Qwen3.8: <|fim_pad|>."""
    for cand in _SENTINEL_PREFS:
        ids = tok(cand, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return cand, ids[0]
    raise ValueError("no single-token sentinel found for this tokenizer")


class SoftTokenObject:
    """One trainable input-embedding vector, injected as a sentinel token."""

    kind = "token"

    #: Why this object REFUSES a position-masked (decode-only) fit: the
    #: intervention is the input embedding of a sentinel token injected
    #: into the PROMPT (right after BOS), so every position it can act on
    #: is a prompt position by construction — there is no assignment of
    #: "steer only the generated tokens" for it. Masking the sentinel out
    #: would leave a raw reserved token in the prompt: neither the method
    #: nor a no-op.
    FIT_MASK_REFUSAL = (
        "the intervention is a soft PROMPT token (a sentinel embedding "
        "injected right after BOS); every position it can act on is a "
        "prompt position, so 'steer only the generated tokens' has no "
        "well-defined action for it — masking the sentinel out leaves a "
        "raw reserved token in the prompt, which is neither the method "
        "nor a no-op")

    def __init__(self, model, tok, sentinel: tuple[str, int] | None = None,
                 init: torch.Tensor | None = None):
        self.sentinel_text, self.sentinel_id = sentinel or find_sentinel(tok)
        emb = model.get_input_embeddings()
        if init is None:
            # Wingate-style neutral init: the mean input embedding (the
            # token_bos arm). The faithful Radevski arm passes the
            # instruction-token-mean init instead (2601.05062 §3; wired in
            # methods._train_token_faithful).
            init = emb.weight.detach().float().mean(0)
        self.param = nn.Parameter(init.detach().float().clone())
        self._emb_module = emb

    @property
    def trainable(self) -> list[nn.Parameter]:
        return [self.param]

    def n_params(self) -> int:
        return self.param.numel()

    def inject(self, rendered_prompt: str) -> str:
        """Insert the sentinel right after the BOS text (or prepend)."""
        for bos in ("<|begin_of_text|>", "<bos>", "<s>"):
            if rendered_prompt.startswith(bos):
                return bos + self.sentinel_text + rendered_prompt[len(bos):]
        return self.sentinel_text + rendered_prompt

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        emb = model.get_input_embeddings()

        def hook(_mod, inputs, output):
            if scale == 0.0:
                # Scale 0 must equal the UNSTEERED model (the invariant every
                # residual arm satisfies bitwise): pass the sentinel's native
                # embedding through rather than zeroing it.
                return output
            ids = inputs[0]
            mask = ids == self.sentinel_id
            if mask.any():
                output = output.clone()
                output[mask] = (scale * self.param).to(output.dtype)
            return output

        h = emb.register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()

    def to_(self, device) -> None:
        self.param.data = self.param.data.to(device)

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind, "param": self.param.detach().float().cpu(),
             "sentinel": self.sentinel_text},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "SoftTokenObject":
        text = blob["sentinel"]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"saved sentinel {text!r} is not a single token")
        obj = cls(model, tok, sentinel=(text, ids[0]))
        obj.param.data = blob["param"].clone()
        return obj


class RankOneLoRAObject(PositionMaskedFit):
    """dW = B A^T on the attention output projection of one layer.

    Residual contribution: scale * B * (A . x). B zero-init => exactly the
    unsteered model at step 0 (matching the vector's zero-init convention).

    Position-masked fits (PositionMaskedFit): o_proj is a position-wise
    linear map, so the weight edit's contribution at position t depends
    only on position t's o_proj input — restricting the delta to the
    handle's mask is EXACT ("the weight edit is active at these positions"),
    with masked positions bit-identical to the unsteered model. apply()
    yields the live mask handle (None in the default all-positions fit) so
    trainers can tie the loss gather to the same mask.
    """

    kind = "lora"

    def __init__(self, model, layer: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.layer = layer
        block = get_decoder_layers(model)[layer]
        # Capability check: hybrid models interleave blocks that have
        # NO attention o_proj — Qwen3.5's linear-attention blocks; NemotronH's
        # Mamba2 and MLP mixers (Nemotron-3-Nano-4B's depth-fraction layer
        # 20 is an MLP block; its attention sits at 12, 17, 24, 32) — name
        # the real problem instead of dying with a bare AttributeError, and
        # resolve through the ONE shared probe (models.attention_oproj, the
        # same resolver oproj_layers filters with). Valid layers untouched.
        oproj = attention_oproj(block)
        if oproj is None:
            capable = [i for i, b in enumerate(get_decoder_layers(model))
                       if attention_oproj(b) is not None]
            raise ValueError(
                f"decoder layer {layer} has no self_attn.o_proj (nor an "
                f"attention-mixer o_proj) ({type(block).__name__} — a "
                "linear-attention / Mamba / MLP block on a hybrid model?): "
                "the rank-1 o_proj LoRA can only target attention layers — "
                f"models.oproj_layers() = {capable}")
        self._module = oproj
        # A dots the o_proj INPUT (num_heads*head_dim) and B is added to its
        # OUTPUT (hidden). These differ on models whose head_dim*n_heads !=
        # hidden_size (gemma-2-9b: o_proj is 4096->3584); sizing both from
        # hidden_size only works on square-o_proj models like Llama and
        # silently mis-shapes A elsewhere. Read the real module dims.
        d_in = self._module.weight.shape[1]
        d_out = self._module.weight.shape[0]
        self.A = nn.Parameter(torch.randn(d_in, generator=g) / d_in**0.5)
        self.B = nn.Parameter(torch.zeros(d_out))

    @property
    def trainable(self) -> list[nn.Parameter]:
        return [self.A, self.B]

    def n_params(self) -> int:
        return self.A.numel() + self.B.numel()

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mask_handle = self._fit_mask_handle(model)

        def hook(_mod, inputs, output):
            x = inputs[0]
            coef = x.to(self.A.dtype) @ self.A  # [..., ]
            delta = coef.unsqueeze(-1) * self.B  # [..., d]
            steered = output + (scale * delta).to(output.dtype)
            return self._fit_masked(mask_handle, steered, output)

        h = self._module.register_forward_hook(hook)
        try:
            yield mask_handle
        finally:
            h.remove()
            if mask_handle is not None:
                mask_handle.remove()

    def to_(self, device) -> None:
        self.A.data = self.A.data.to(device)
        self.B.data = self.B.data.to(device)

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind, "A": self.A.detach().float().cpu(),
             "B": self.B.detach().float().cpu(), "layer": self.layer,
             # The FIT's mask leg is part of the artifact's identity (the
             # DiffMeanObject/LoReFT precedent, 2026-08-21): before this key
             # existed a response-fit LoRA re-hydrated ALL-POSITION and would
             # silently run the all-token method under a decode-only label.
             "positions": self.fit_positions,
             "response_header_len": self.fit_response_header_len},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "RankOneLoRAObject":
        obj = cls(model, layer=int(blob["layer"]))
        obj.A.data = blob["A"].clone()
        obj.B.data = blob["B"].clone()
        # Backward-compatible fit-leg restore (the DiffMeanObject pattern):
        # no positions key -> the historical all-token fit, byte-identical.
        if blob.get("positions", "all") != "all":
            obj.set_fit_positions(blob["positions"],
                                  blob.get("response_header_len"))
        return obj


def train_object_kl(
    model,
    tok,
    obj,
    examples: list[DistillExample],
    *,
    lr: float = 5e-3,
    weight_decay: float = 0.01,
    epochs: int = 6,
    batch_size: int = 2,
    max_len: int = 768,
    temperature: float = 1.0,
    loss_scale: float = 1.0,
    loss: str = "soft",
    seed: int = 0,
    log_fn=None,
    positions: str = "all",
    response_header_len: int | None = None,
    reg_coeff: float = 0.0,
    teacher_prefix_cache: bool = False,
) -> list[float]:
    """Same forward-KL objective and schedule as train_steered_lm(loss=
    'soft'); only the intervention object differs. Teacher logits are
    computed with the object INACTIVE (plain prompted model).

    ``teacher_prefix_cache``: the teacher prompts share their rendered
    prefix (chat header + system=SI), so under the flag the prefix runs
    through the model ONCE and every GRADIENT-TRAINING teacher forward
    replays it as past_key_values (:mod:`.prefix_cache` — semantically
    exact, fp-tolerance-only difference on the training legs; the two-SI
    config's ``teacher_prefix_cache: false`` is the escape hatch). The
    startup probe stays AUTHORITATIVE AND UNCACHED, flag or no flag (see
    ``_uncached_teacher_logits`` below and the cache-call-count test in
    tests/test_prefix_cache.py). Derived from the fit batches, exactly as
    lockstep derives its group cache, so solo-vs-lockstep stays bitwise
    under the same flag. Default ``False`` keeps every other caller
    byte-identical.
    Teacher legs only — the student forward carries the gradients and is
    untouched; the hard loss consumes no teacher logits, so the flag is a
    no-op there.

    ``reg_coeff`` > 0 adds ``reg_coeff * obj.gate_reg_term(...)`` to every
    step's loss — the PSR anti-dead-gate regularizer (train_psr passes its
    own 0.1 default through, the 2026-08-20 confound fix giving both PSR
    objectives the same protection). The object MUST expose
    ``gate_reg_term`` (PSRObject's shared assembly) or this refuses
    loudly; the default 0.0 leaves every other arm's training
    byte-identical (no term is computed at all). The startup probe and
    its no-op floor stay PURE divergence: the term is identically zero at
    the wide-open gate init, and a floor "with the object inactive" has
    no gates to pool — so the floor-ratio guard's calibration is
    untouched.

    `temperature` softens both distributions; `loss_scale` multiplies the
    KL. The Radevski (2601.05062) faithful Steering-Token recipe passes
    temperature=10, loss_scale=100 (the distillation ×T² gradient
    correction). Defaults reproduce the original T=1 objective exactly.

    ``positions``/``response_header_len`` select the fit's mask leg
    (hooks.FIT_POSITION_MODES): under ``"response"`` the object's hooks
    steer only the boundary-onward positions (header-steered when
    ``response_header_len`` is set — H is derived ONCE by the caller, never
    here) and the loss gathers exactly the completion positions that mask
    covers, from the SAME handle the hooks consumed (``loss_gather_masks``
    — the mask→loss wiring). An object that exposes no fit mask
    (``set_fit_positions``) REFUSES the masked fit loudly: a silent
    all-token fit under a decode-only label is exactly the artifact the
    wiring exists to prevent. The default ``"all"`` path is byte-identical
    to the historical behavior."""
    if reg_coeff and not hasattr(obj, "gate_reg_term"):
        raise ValueError(
            f"reg_coeff={reg_coeff} but {type(obj).__name__} (kind "
            f"{obj.kind!r}) exposes no gate_reg_term — the anti-dead-gate "
            "regularizer is a PSR-gate mechanism; do not pass reg_coeff "
            "for an arm without a gate.")
    if positions != "all":
        set_fit = getattr(obj, "set_fit_positions", None)
        if set_fit is None:
            reason = getattr(
                obj, "FIT_MASK_REFUSAL",
                "its apply() exposes no fit position mask "
                "(no set_fit_positions)")
            raise ValueError(
                f"{type(obj).__name__} (kind {obj.kind!r}) cannot honor a "
                f"positions={positions!r} (decode-only) fit: {reason} — "
                "never fit it all-token under a decode-only label.")
        set_fit(positions, response_header_len)

    def _live_handle(ctx_value):
        return ctx_value if isinstance(ctx_value, SteeringHandle) else None

    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in obj.trainable:
        p.data = p.data.to(device)
    # weight_decay exposed 2026-08-27 (re-tune): the default preserves
    # AdamW's own 0.01, which every prior fit silently trained under.
    opt = torch.optim.AdamW(obj.trainable, lr=lr, weight_decay=weight_decay)

    encoded = []
    _kl_prog = TokenizeProgress(len(examples), label="object_kl train",
                                max_len=max_len)
    for ex in examples:
        # inject() is identity for every non-token object, so applying it
        # unconditionally keeps one code path for all kinds (after-BOS
        # token, appended token, everything else).
        ex = DistillExample(
            prompt_with_si=ex.prompt_with_si,
            prompt_without=obj.inject(ex.prompt_without),
            completion=ex.completion,
            instruction_idx=ex.instruction_idx,
        )
        encoded.append(_encode_aligned(tok, ex, max_len))
        _kl_prog.tick(len(encoded))
    _kl_prog.done()
    batches: list[tuple[Batch, Batch]] = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        batches.append(
            (collate([t for t, _ in chunk], tok.pad_token_id),
             collate([s for _, s in chunk], tok.pad_token_id))
        )

    # Teacher prefix cache (see the docstring): one prefix forward for the
    # whole fit, replayed by every teacher leg below. The hard loss never
    # consumes teacher logits, so nothing is built for it.
    tpc = None
    if teacher_prefix_cache and loss != "hard":
        tpc = prefix_cache_for_batches(model, [t for t, _ in batches])

    def _uncached_teacher_logits(teacher_b: Batch) -> torch.Tensor:
        """The plain full forward + completion gather — the AUTHORITATIVE
        assembly for the startup probe (decision-bearing convention:
        probes never read cached kernels, so their numbers are identical
        whether or not the flag is on)."""
        with torch.no_grad():
            return _gather_completion_logits(
                model(input_ids=teacher_b.input_ids,
                      attention_mask=teacher_b.attention_mask).logits,
                teacher_b,
            )

    def _teacher_completion_logits(teacher_b: Batch) -> torch.Tensor:
        """The completion-gathered teacher logits for a TRAINING step —
        prefix-cached when the cache exists, else the historical full
        forward, byte-identical. Never used by the probe."""
        if tpc is not None:
            return tpc.completion_logits(model, teacher_b)
        return _uncached_teacher_logits(teacher_b)

    # Startup sanity check — two no-grad passes over the FIRST batch (in
    # construction order): the as-initialized loss vs the same batch with
    # the object INACTIVE (the no-op floor). Warns loudly when the init
    # starts far off the unsteered model (how the diffmean-init all-layer
    # PSR silently gate-died at step 1); mutates nothing and draws no RNG,
    # so trajectories — and the lockstep bitwise contract — are untouched.
    # Masked fits measure the startup probe (and its no-op floor) on the
    # SAME position support the loss will train on — the lockstep
    # _probe_pair convention. The gather needs no forward: the handle's
    # mask under per-row boundaries is a pure function of the batch.
    probe_gather = probe_keep = None
    if positions != "all":
        _t0, _s0 = batches[0]
        _s0 = _s0.to(device)
        _ph = obj._fit_mask_handle(model)
        try:
            _ph.set_prompt_lens(
                steer_prompt_boundaries(_s0, _ph.response_header_len))
            probe_gather, _tg, probe_keep = loss_gather_masks(_ph, _s0)
        finally:
            _ph.remove()

    def _batch0_loss(active: bool) -> float:
        teacher_b, student_b = batches[0]
        teacher_b, student_b = teacher_b.to(device), student_b.to(device)
        with torch.no_grad():
            with obj.apply(model, scale=1.0) if active else nullcontext() \
                    as ctx_value:
                handle = _live_handle(ctx_value) if active else None
                if handle is not None and handle.positions in (
                        "response", "prompt"):
                    handle.set_prompt_lens(steer_prompt_boundaries(
                        student_b, handle.response_header_len))
                s_logits = _gather_completion_logits(
                    model(input_ids=student_b.input_ids,
                          attention_mask=student_b.attention_mask).logits,
                    student_b,
                    gather=probe_gather,
                )
            if loss == "hard":
                labels = student_b.labels[:, 1:]
                targets = (labels[probe_gather] if probe_gather is not None
                           else labels[labels != -100])
                return float(loss_scale
                             * F.cross_entropy(s_logits.float(), targets))
            # AUTHORITATIVE AND UNCACHED — the probe convention.
            t_logits = _uncached_teacher_logits(teacher_b)
            if probe_keep is not None and not bool(probe_keep.all()):
                t_logits = t_logits[probe_keep]
            t_logp = torch.log_softmax(t_logits.float() / temperature, -1)
            s_logp = torch.log_softmax(s_logits.float() / temperature, -1)
            return float(loss_scale * F.kl_div(s_logp, t_logp,
                                               log_target=True,
                                               reduction="batchmean"))

    startup_loss_check(_batch0_loss(True), _batch0_loss(False),
                       tag=f"object_{obj.kind}", log_fn=log_fn)

    history: list[float] = []
    step = 0
    for epoch in range(epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        for bi in order:
            teacher_b, student_b = batches[bi]
            teacher_b, student_b = teacher_b.to(device), student_b.to(device)
            # The mask→loss wiring (lockstep's _masked_student_pass shape):
            # a masked object's apply() yields its live mask handle — set
            # per-row boundaries BEFORE the forward (never derived from the
            # padded batch width) and gather exactly the completion
            # positions the hook's mask covers. handle is None on the
            # default all-token leg, which stays byte-identical.
            s_gather = keep = None
            reg = None
            with obj.apply(model, scale=1.0) as ctx_value:
                handle = _live_handle(ctx_value)
                if handle is not None and handle.positions in (
                        "response", "prompt"):
                    handle.set_prompt_lens(steer_prompt_boundaries(
                        student_b, handle.response_header_len))
                logits = model(input_ids=student_b.input_ids,
                               attention_mask=student_b.attention_mask).logits
                if handle is not None:
                    s_gather, _t_gather, keep = loss_gather_masks(
                        handle, student_b)
                if reg_coeff:
                    # Inside the ctx (the gate stash clears at exit), via
                    # the object's ONE shared assembly — train_lockstep
                    # computes the identical term the identical way, which
                    # is what keeps the solo-vs-lockstep bitwise contract
                    # green for the regularized PSR KL arms.
                    reg = obj.gate_reg_term(student_b.attention_mask,
                                            handle)
            s_logits = _gather_completion_logits(logits, student_b,
                                                 gather=s_gather)
            if loss == "hard":
                # LL / hard-token cross-entropy on the teacher completion
                # tokens (HyperSteer Eq. 4) — the one-hot limit of the soft
                # forward-KL branch below; no teacher forward needed.
                labels = student_b.labels[:, 1:]
                targets = (labels[s_gather] if s_gather is not None
                           else labels[labels != -100])
                step_loss = loss_scale * F.cross_entropy(s_logits.float(),
                                                         targets)
                loss_tag = "hard_loss"
            else:
                t_logits = _teacher_completion_logits(teacher_b)
                if keep is not None and not bool(keep.all()):
                    # Sub-select the teacher logits to the same completion
                    # indices the student's mask kept (plain decode-only
                    # drops the first completion logit; header-steered
                    # keeps everything and lands here with keep=None).
                    t_logits = t_logits[keep]
                t_logp = torch.log_softmax(t_logits.float() / temperature, -1)
                s_logp = torch.log_softmax(s_logits.float() / temperature, -1)
                step_loss = loss_scale * F.kl_div(s_logp, t_logp,
                                                  log_target=True,
                                                  reduction="batchmean")
                loss_tag = "kl_loss"
            if reg is not None:
                step_loss = step_loss + reg_coeff * reg
            opt.zero_grad()
            step_loss.backward()
            opt.step()
            step += 1
            loss_val = float(step_loss.detach())
            history.append(loss_val)
            if log_fn:
                rec = {f"object_{obj.kind}/{loss_tag}": loss_val,
                       "epoch": epoch}
                if reg is not None:
                    rec[f"object_{obj.kind}/gate_reg"] = float(reg.detach())
                log_fn(rec, step)
    return history


def _chunk_check_slice(batches: list[Batch], max_len: int) -> Batch:
    """The trainer's bounded cross-check input: the FIRST row (the batches
    are length-sorted, shortest first) that has a CE target within its
    first ``max_len`` positions, cut to ``min(max_len, its real length)``
    — the padding drops with the tail. The cut is NEVER extended past the
    cap: a row whose masked prefix exceeds it is skipped (an extension
    could materialise near-full logits before the loop even starts), and a
    pool with no qualifying row is refused, so the check's full-logits
    allocation is bounded by ``max_len`` positions by construction. Scans
    batches in order and stops at the first hit (batch 0 row 0 for every
    real pool); one small device sync per scanned batch."""
    cap = int(max_len)
    for b in batches:
        n_real = b.attention_mask.sum(dim=1)                 # [B]
        cut = torch.clamp(n_real, max=cap)                    # per-row cut
        # labels[:, 1:] index j is the target at position j+1; it is inside
        # a cut of c positions iff j + 2 <= c.
        tgt = b.labels[:, 1:cap] != -100                      # [B, <=cap-1]
        if tgt.shape[1] == 0:
            continue
        j = torch.arange(tgt.shape[1], device=tgt.device)
        inside = tgt & ((j[None, :] + 2) <= cut[:, None])
        hit = inside.any(dim=1).nonzero()
        if hit.numel() == 0:
            continue
        i = int(hit[0])
        c = int(cut[i])
        return Batch(b.input_ids[i:i + 1, :c], b.attention_mask[i:i + 1, :c],
                     b.prompt_lens[i:i + 1], b.labels[i:i + 1, :c])
    raise ValueError(
        f"chunked_ce cross-check: no row has a CE target within its first "
        f"{cap} positions (every masked prefix is longer than "
        "chunk_check_max_len) — the bounded startup check cannot run. Raise "
        "chunk_check_max_len (it caps the check's full-logits allocation) "
        "or fix the pool.")


def shuffled_epoch_order(rng: random.Random, n_batches: int) -> list[int]:
    """The trainer's per-epoch micro-batch order: ``range(n_batches)``
    shuffled IN PLACE by ``rng`` — the trainer's own ``random.Random(seed)``
    stream, on which NO draw precedes the first epoch's shuffle (a resume
    restores the saved state instead), so epoch 0's order is a pure
    function of ``seed`` and ``n_batches``. Shared with the realized-batch
    validator (sisteer.extraction.realized_batch): what it reproduces IS
    what the trainer ran, by construction rather than by re-derivation."""
    order = list(range(int(n_batches)))
    rng.shuffle(order)
    return order


def group_row_counts(n_rows, order, accum_steps: int,
                     skip: int = 0) -> list[int]:
    """Rows per optimizer step (accumulation group) along ``order``: groups
    of ``accum_steps`` consecutive micro-batches starting at ``skip``,
    never spanning epochs — the ragged final group is its own smaller step
    (contract (i)). The micro-batch carrying the row deficit (the LAST
    length-sorted chunk when ``n_docs % batch_size != 0``) lands wherever
    the shuffled order put it, so the deficit sits in ANY group and only
    the last group is short by COUNT. This is ``fit_record.
    realized_eff_batch``'s per-group arithmetic, in one place."""
    acc = int(accum_steps)
    return [sum(int(n_rows[b]) for b in order[g0:g0 + acc])
            for g0 in range(int(skip), len(order), acc)]


def train_object_ce(
    model,
    tok,
    obj,
    examples: list[DistillExample],
    *,
    lr: float,
    weight_decay: float = 0.01,
    epochs: int,
    batch_size: int,
    max_len: int,
    seed: int = 0,
    log_fn=None,
    log_tag: str = "object_ce",
    accum_steps: int = 1,
    lr_schedule: str | None = None,
    warmup_steps: int | None = None,
    max_grad_norm: float | None = None,
    heldout: list[DistillExample] | None = None,
    early_stop: EarlyStopSpec | None = None,
    diagnostics: dict | None = None,
    reg_coeff: float = 0.0,
    checkpoint_every: int | None = None,
    checkpoint_dir=None,
    checkpoint_key: str | None = None,
    positions: str = "all",
    response_header_len: int | None = None,
    loss_impl: str = "full",
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    chunk_check_rtol: float = 1e-2,
    chunk_check_max_len: int = 1024,
    max_steps: int | None = None,
    replay_exempt_epoch: int | None = None,
    epoch_cb=None,
) -> list[float]:
    """Solo hard-target cross-entropy trainer on DOCUMENT tokens — plain
    LM CE over the completion slot, no teacher forward of any kind. The
    intervention object trains in place; returns the per-optimizer-step
    loss history.

    Deliberately a NEW trainer next to :func:`train_object_kl` (option (c)
    of the plan): extending the KL loop would thread heldout/early-stop/
    accumulation/schedule/checkpoint state through a function whose
    byte-identical behavior is a standing contract, and lockstep is
    teacher-sharing machinery CE has no use for. The pieces below are
    REUSED, not re-implemented: ``_encode_aligned``/``collate``/
    ``_gather_completion_logits`` (encoding + mask→loss),
    ``startup_loss_check``, ``EarlyStopSpec``/``EarlyStopper``, the psr
    mse loop's schedule/accumulation/checkpoint conventions and
    ``fitckpt``.

    The four contract items (not implementation latitude):

    - **(i) token-share accumulation weighting** — each optimizer step
      spans ``accum_steps`` consecutive micro-batches; each micro-batch's
      CE is weighted by its COMPLETION-TARGET-token share of the
      accumulation GROUP (``n_tok[b]/grp_tok`` — the audited PSR mse-loop
      rule, psr.py's ``w_mse``) and the PSR gate regularizer by its ROW
      share, so the accumulated gradient EQUALS one true concatenated
      batch's and the objective is invariant to bucketing, batch width and
      ragged tails. NEVER a flat ``1/accum_steps``. Accumulation groups
      never span epochs: the partial group at epoch end is its own
      (smaller) optimizer step, which keeps tiny cells consistent with the
      driver's warmup arithmetic (``epochs × ceil(ceil(n/bs)/accum)``)
      with no special-case clamp. Clipping fires ONCE per optimizer step
      on the ACCUMULATED gradient.
    - **(ii) held-out CE, micro-averaged, + mandatory early stopping** —
      held-out CE is ``sum(token NLL) / sum(target tokens)`` over the
      WHOLE pool (never a mean of batch means, which up-weights short docs
      and makes the metric depend on bucketing), evaluated no-grad after
      every epoch; ``EarlyStopSpec`` patience + best-epoch snapshot
      restore. ``heldout_noop_ce`` is computed ONCE before epoch 1 with
      the object INACTIVE (``nullcontext`` — exactly the unmodified base
      model): the bare-base denominator the CE floor ratio divides by.
    - **(iii) hook lifetime** — the forward, the loss assembly AND
      ``backward()`` all run inside ``obj.apply(model, scale=1.0)``: qwen
      trains under non-reentrant gradient checkpointing, which RECOMPUTES
      the forward during backward, and a recompute without the hooks is a
      different graph (the 2026-08-27 psr diagnosis; pinned for both CE
      arms by tests/test_ckpt_grad_parity.py).
    - **(iv) checkpoints at completed optimizer-group boundaries only**
      (``fitckpt`` conventions): trainable params, optimizer state,
      schedule position, epoch/step counters, the epoch's shuffled order +
      RNG states, the held-out curve, ``heldout_noop_ce`` and the FULL
      stopper state (best value, patience counter, best snapshot) all ride
      the checkpoint, so an interrupted fit resumes bitwise on CPU fp32.

    Also enforced here: a per-example ``max_len`` assertion (prompt +
    completion + EOS must FIT — this trainer never relies on
    ``_encode_aligned``'s silent ``comp[:avail]`` truncation, which would
    quietly shrink a cell's realized token budget); a non-finite train or
    held-out loss raises :class:`~sisteer.extraction.fitmon.TrainingDiverged`
    (never train through NaN); ``reg_coeff`` requires a gate-bearing
    object (``gate_reg_term``), refused otherwise; ``positions`` must be
    ``"all"`` — the document fits steer every position by design, and no
    masked-fit wiring exists on this path (a decode-only label over an
    all-token fit is exactly the artifact the refusal prevents).

    Empty-prompt/BOS edges are inherited from ``collate``'s labels + the
    ``labels[:, 1:]`` shift: a tokenizer that maps ``""`` to ``[BOS]``
    supervises every completion token; one that maps it to ``[]``
    (prompt_len 0) drops only position 0's token from the targets — the
    standard LM convention. Both shapes are pinned by tests.

    ``loss_impl`` selects HOW the CE is computed, never WHAT:
    ``"full"`` (default — every published fit) materialises the
    ``[B, S, V]`` logits and gathers; ``"chunked_ce"`` runs the SAME
    forward through the model's own ``forward`` (every steering hook, on
    decoder layers and on the top-level model alike, fires as before),
    captures the body's last hidden state, gathers the completion
    positions FIRST and applies ``lm_head`` (+ the model's final logit
    softcap when its config declares one) ``chunk_tokens`` target rows at
    a time, fp32 loss per chunk, one body backward — see
    :mod:`~sisteer.extraction.chunked_ce`. Same objective, same
    token-share weighting, same hook lifetime (chunk loop AND backward
    inside ``apply()``); peak memory drops by the whole-sequence logits.
    On entry the chunked trainer scores ONE bounded slice — the first
    (shortest) row with a CE target inside its first ``chunk_check_max_len``
    positions, cut there (never extended; a pool with no such row is
    refused) — BOTH ways (no grad) and refuses when the relative difference
    exceeds ``chunk_check_rtol``: a wiring fault (wrong body module, a
    missing softcap) shows up as ~1e-1 there, bf16 kernel-tiling noise as
    <~1e-3. The slice bounds the check's one full-logits allocation
    (1024 x 262k x 10 B ~ 2.7 GB) below a single chunk's transient, so
    the check can never be the memory peak nor OOM at a batch size the
    chunked loop fits — including a worst-case memory probe whose batch 0
    is itself ~8k tokens wide. The implementation is a fit-identity
    dimension (``diagnostics["loss_impl"]``; the driver stamps it into the
    fit fingerprint) because bf16 numerics differ at the kernel level even
    though the mathematics is identical.

    CUDA peak-memory statistics are reset right before the first optimizer
    group: the startup passes (bare-base held-out CE, the cross-check) are
    no-grad transients strictly below the training loop's footprint, and
    the memory probe / bench read ``max_memory_*`` AFTER this call as the
    fit's ceiling — the LOOP's, which is what a production cell runs at.
    Numerics are untouched (statistics only).

    ``max_steps`` bounds the run at that many OPTIMIZER steps — the
    throughput bench / smoke knob (``diagnostics["stopped_by_max_steps"]``
    says it fired). It is refused together with ``early_stop`` or
    checkpoints: a bounded run is never a publishable fit.

    **Data-parallel (group-sharded DDP).**
    Under an ambient :mod:`~sisteer.extraction.ddp` context of ``N`` ranks
    (the arm launched by ``torchrun --nproc_per_node=N``) this loop computes
    EXACTLY the single-process fit, split by micro-batch: every rank holds
    the full length-bucketed batch list, walks the SAME per-epoch shuffled
    order and the SAME accumulation groups, and forwards + backwards only its
    strided share ``group[rank::N]`` of each group's micro-batches (the
    ``accum_steps`` micro-batches of one optimizer step) with the GLOBAL
    token-share weights (``n_tok[b] / grp_tok`` over the whole group), so the
    union of rows per optimizer step is the single-GPU step's and the
    per-rank partial gradients SUM to the single-GPU accumulated gradient.
    That sum is ONE flat all-reduce over the object's trainable (adapter)
    parameters only (:func:`ddp.allreduce_grads_and_stats` — the frozen base
    never enters a collective, so nothing wraps the base model's graph in
    DDP buckets); the loss shares are summed in fp64; every rank then clips and
    steps its own optimizer copy on the identical gradient, so replicas stay
    bit-equal (:func:`ddp.assert_replicas_equal` before the return). The
    effective batch, the step count, the schedule and the realized-batch
    vector are UNCHANGED — per-rank accumulation is ``accum_steps / N``, and
    ``accum_steps % N != 0`` is refused by name (an unequal split would still
    be exact, but the plan's "effective batch 64 ÷ (N × micro_batch)"
    accounting names equal shares). Held-out CE shards the held-out batches
    the same way and all-reduces ``(sum NLL, target count)`` in fp64, so the
    stopper rules identically on every rank. Checkpoints are written by rank
    0 only and resumed by every rank (the same file, the same state); the
    startup checks run redundantly on every rank (no collective). RNG: the
    epoch shuffle and ``torch.manual_seed`` use the shared run seed on every
    rank BY DESIGN — the order and the replicas must agree — and the loop
    itself draws no random numbers (the model is in eval mode, no dropout;
    the object's init is seeded before this call), so rank identity enters
    only through the deterministic shard. ``diagnostics["fit_width"]`` = N.
    With no context every helper is an identity and this path is
    byte-identical to before the port.

    ``replay_exempt_epoch`` (the continuation stopper rule
    ``replay_exempt``): the 0-based index of the epoch a
    CONTINUATION resumed INTO from a retained mid-epoch checkpoint. The
    remainder of that epoch (micro-batches after the checkpoint's
    ``batches_done``, up to the epoch end) is a PARTIAL epoch trained under
    a freshly restarted lr; its end-of-epoch held-out value is computed,
    logged (``<tag>/heldout_loss`` with ``<tag>/replayed_partial`` = 1)
    and kept in the curve, but it does NOT update the early-stopping
    state — neither a strike nor a new best (``EarlyStopper.
    observe_exempt``): the seated incumbent stands, and the first FULL
    post-restart epoch is where patience starts counting. Only the epoch
    with that index is exempt, and only when this process actually resumed
    into it: the trainer REFUSES the kwarg without checkpoints, without a
    checkpoint to resume, or when the resumed checkpoint predates the named
    epoch (an exemption on a fresh fit would silence a real epoch). A
    successor that resumes a later checkpoint (past the replayed epoch)
    applies no exemption and inherits the record from the checkpoint
    (``replayed_partial_epochs``, 1-based, rides every checkpoint and the
    diagnostics). Every epoch under the legacy rule (the kwarg absent) is
    ruled on as before — the trajectory is untouched either way; only the
    stopper's bookkeeping differs.
    """
    from .fitmon import TrainingDiverged

    if positions != "all":
        raise ValueError(
            f"train_object_ce trains the all-position leg only, got "
            f"positions={positions!r} — the document-CE fits steer every "
            "position by design (spec: positions_at_fit='all') and this "
            "trainer has no masked-fit wiring; refusing to fit an "
            "all-token object under a decode-only label.")
    if response_header_len is not None:
        raise ValueError(
            "train_object_ce takes no response_header_len — there is no "
            "masked leg on this path (positions='all' only).")
    if reg_coeff and not hasattr(obj, "gate_reg_term"):
        raise ValueError(
            f"reg_coeff={reg_coeff} but {type(obj).__name__} (kind "
            f"{obj.kind!r}) exposes no gate_reg_term — the anti-dead-gate "
            "regularizer is a PSR-gate mechanism; do not pass reg_coeff "
            "for an arm without a gate.")
    if early_stop is not None and not heldout:
        raise ValueError(
            "early_stop without heldout — the stopper would have no "
            "held-out curve to rule on and the fit would silently run the "
            "full epoch budget wearing an early-stop request; pass the "
            "held-out examples or drop the spec.")
    if replay_exempt_epoch is not None:
        replay_exempt_epoch = int(replay_exempt_epoch)
        if replay_exempt_epoch < 0:
            raise ValueError(
                f"replay_exempt_epoch must be a 0-based epoch index, got "
                f"{replay_exempt_epoch!r}")
        if checkpoint_every is None:
            raise ValueError(
                "replay_exempt_epoch without checkpoints — the exemption "
                "names the epoch a CONTINUATION resumed into from a retained "
                "checkpoint; a fit that cannot resume has no replayed epoch "
                "(R2-D77)")
        if not heldout:
            raise ValueError(
                "replay_exempt_epoch without heldout — there is no "
                "early-stopping state to exempt the replayed epoch from")
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs!r}")
    accum_steps = int(accum_steps)
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps!r}")
    checkpoint_every = resolve_checkpoint_kwargs(
        checkpoint_every, checkpoint_dir, checkpoint_key)
    if lr_schedule is None:
        lr_schedule = "warmup_cosine"
    if lr_schedule not in ("warmup_cosine", "constant"):
        raise ValueError(
            f"lr_schedule must be 'warmup_cosine' or 'constant', got "
            f"{lr_schedule!r}")
    if max_grad_norm is None:
        max_grad_norm = 1.0
    if loss_impl not in LOSS_IMPLS:
        raise ValueError(
            f"loss_impl must be one of {LOSS_IMPLS}, got {loss_impl!r}")
    chunked = loss_impl == "chunked_ce"
    chunk_tokens = int(chunk_tokens)
    if chunked and chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be >= 1, got {chunk_tokens!r}")
    chunk_check_max_len = int(chunk_check_max_len)
    if chunked and chunk_check_max_len < 2:
        raise ValueError("chunk_check_max_len must be >= 2 (one target), "
                         f"got {chunk_check_max_len!r}")
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps!r}")
        if early_stop is not None or checkpoint_every is not None:
            raise ValueError(
                "max_steps is a bounded-run knob (throughput bench / smoke) "
                "and cannot combine with early_stop or checkpointing — a "
                "step-capped run is never a publishable fit.")

    # The data-parallel context (docstring): N ranks share every
    # optimizer step's micro-batches; identity helpers at width 1.
    from . import ddp as _ddp

    par = _ddp.active()
    world = int(par.world_size) if (par is not None and par.enabled) else 1
    rank = int(par.rank) if (par is not None and par.enabled) else 0
    is_main = rank == 0
    if world > 1 and accum_steps % world != 0:
        raise ValueError(
            f"{log_tag}: fit_width={world} does not divide accum_steps="
            f"{accum_steps} — a data-parallel fit splits each optimizer "
            f"step's {accum_steps} micro-batches ({accum_steps} x "
            f"{int(batch_size)} rows, the effective batch) into EQUAL "
            f"per-rank accumulation shares of {accum_steps}/{world}; pick a "
            "fit width that divides accum_steps (the effective batch ÷ "
            "the micro-batch), or run at width 1.")
    if world > 1:
        print(f"[{log_tag}] data-parallel fit: width {world}, this rank "
              f"{rank}; per-rank accumulation {accum_steps // world} of "
              f"{accum_steps} micro-batches per optimizer step (effective "
              f"batch {accum_steps * int(batch_size)} rows, unchanged)",
              flush=True)

    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in obj.trainable:
        p.data = p.data.to(device)
    opt = torch.optim.AdamW(obj.trainable, lr=lr, weight_decay=weight_decay)
    # The chunked path's fixed pieces, resolved ONCE (and after the freeze
    # above, which is what lm_head_of's frozen-head assertion rules on).
    _lm_head = lm_head_of(model) if chunked else None
    _softcap = final_logit_softcap(model) if chunked else None
    if chunked:
        print(f"[{log_tag}] loss_impl=chunked_ce chunk_tokens={chunk_tokens} "
              f"final_logit_softcap={_softcap}", flush=True)

    def _encode_checked(exs, label: str) -> list:
        """Student-leg encodings with the per-example max_len REFUSAL:
        ``_encode_aligned`` truncates over-length completions
        SILENTLY (``comp[:avail]``), which would quietly shrink the cell's
        realized token budget — assert the fit first, then encode."""
        out = []
        prog = TokenizeProgress(len(exs), label=label, max_len=max_len)
        eos_n = 1 if tok.eos_token_id is not None else 0
        for i, ex in enumerate(exs):
            # inject() is identity for every non-token object (the
            # train_object_kl convention) — the CE arms' objects never
            # rewrite the prompt, but a soft-token object trained through
            # this loop must still carry its sentinel.
            ex = DistillExample(
                prompt_with_si=ex.prompt_with_si,
                prompt_without=obj.inject(ex.prompt_without),
                completion=ex.completion,
                instruction_idx=ex.instruction_idx,
            )
            n_comp = len(cached_ids(tok, ex.completion, False)) + eos_n
            if n_comp - eos_n < 1:
                raise ValueError(
                    f"{label}: example {i} has an EMPTY completion — "
                    "nothing to supervise (the doc sampler asserts "
                    "non-empty trained text; a pool row without it is "
                    "broken).")
            n_prompt = max(len(cached_ids(tok, ex.prompt_with_si, True)),
                           len(cached_ids(tok, ex.prompt_without, True)))
            if n_prompt + n_comp > max_len:
                raise ValueError(
                    f"{label}: example {i} needs {n_prompt} prompt + "
                    f"{n_comp} completion(+EOS) = {n_prompt + n_comp} "
                    f"tokens > max_len={max_len} — refusing the silent "
                    "comp[:avail] truncation that would quietly shrink "
                    "this cell's realized training budget. Raise max_len "
                    "(the staging preflight should have caught this row).")
            out.append(_encode_aligned(tok, ex, max_len)[1])  # student leg
            prog.tick(i + 1)
        prog.done()
        return out

    encoded = _encode_checked(examples, f"{log_tag} train")
    # Length-bucketed batches (the production fitmon.length_sorted
    # rationale): sort by encoded length before chunking so every batch
    # pads to near-uniform width. The token-share accumulation makes the
    # OBJECTIVE invariant to this ordering; only throughput changes.
    encoded = sorted(encoded, key=lambda e: len(e.ids))
    batches: list[Batch] = []
    for i in range(0, len(encoded), int(batch_size)):
        chunk = encoded[i:i + int(batch_size)]
        batches.append(collate(chunk, tok.pad_token_id).to(device))
    # Per-batch loss denominators, known before any forward: CE targets are
    # labels[:, 1:] != -100 (the shifted count — one fewer than the raw
    # label count for a prompt_len-0 row, the BOS-less empty-prompt edge).
    n_tok = [int((b.labels[:, 1:] != -100).sum()) for b in batches]
    n_rows = [int(b.input_ids.shape[0]) for b in batches]
    # Throughput accounting (gpu-util skill): REAL (non-pad) tokens the
    # body encodes and the PADDED width it actually computes, per batch —
    # logged per optimizer step so tok/s and the pad fraction are read
    # from the trainer's own record, never inferred from step counts.
    n_real = [int(b.attention_mask.sum()) for b in batches]
    n_padded = [int(b.input_ids.numel()) for b in batches]
    if any(n == 0 for n in n_tok):
        raise ValueError(
            f"{log_tag}: a batch has ZERO CE targets after the label "
            "shift — refusing to train on nothing.")

    h_batches: list[Batch] = []
    if heldout:
        h_encoded = _encode_checked(heldout, f"{log_tag} held-out")
        h_encoded = sorted(h_encoded, key=lambda e: len(e.ids))
        for i in range(0, len(h_encoded), int(batch_size)):
            chunk = h_encoded[i:i + int(batch_size)]
            h_batches.append(collate(chunk, tok.pad_token_id).to(device))

    def _batch_ce(batch: Batch, *, active: bool,
                  impl: str | None = None) -> tuple:
        """(summed NLL over CE targets, target count) for one batch, one
        no-grad forward, object active vs INACTIVE (nullcontext — the
        exact unmodified base model). ``impl`` overrides the trainer's
        ``loss_impl`` for the startup cross-check only."""
        impl = loss_impl if impl is None else impl
        with torch.no_grad():
            with obj.apply(model, scale=1.0) if active else nullcontext():
                if impl == "chunked_ce":
                    h = forward_last_hidden(model, batch.input_ids,
                                            batch.attention_mask)
                    h_sel, targets = completion_hidden(h, batch)
                    nll_t, _g = chunked_ce(_lm_head, _softcap, h_sel,
                                           targets, chunk_tokens=chunk_tokens)
                    return float(nll_t), int(targets.numel())
                logits = model(input_ids=batch.input_ids,
                               attention_mask=batch.attention_mask).logits
        s_logits = _gather_completion_logits(logits, batch)
        labels = batch.labels[:, 1:]
        targets = labels[labels != -100]
        nll = float(F.cross_entropy(s_logits.float(), targets,
                                    reduction="sum"))
        return nll, int(targets.numel())

    def _heldout_ce(active: bool) -> float:
        """Micro-averaged held-out CE: sum(token NLL) / sum(target tokens)
        over the WHOLE pool — never a mean of batch means (contract (ii))."""
        tot_nll = 0.0
        tot_tok = 0
        # Data-parallel: this rank's strided share of the held-out batches,
        # then ONE fp64 all-reduce of (sum NLL, target count) — every rank
        # reads the identical CE, so the stopper rules identically
        # everywhere. Identity (the whole list, no collective) at width 1.
        for hb in _ddp.shard_items(h_batches, par):
            nll, n = _batch_ce(hb, active=active)
            tot_nll += nll
            tot_tok += n
        tot_nll, tot_tok_f = _ddp.allreduce_vec([tot_nll, float(tot_tok)],
                                                par)
        tot_tok = int(round(tot_tok_f))
        ce = tot_nll / max(tot_tok, 1)
        if not math.isfinite(ce):
            raise TrainingDiverged(
                f"{log_tag}: non-finite held-out CE ({ce!r}) — refusing "
                "to keep training through a diverged evaluation.")
        return ce

    # Startup sanity check (the train_object_kl convention): batch-0 CE
    # as-initialized vs the same batch with the object INACTIVE. Zero-init
    # objects start exactly at the floor; mutates nothing, draws no RNG.
    _b0_act, _b0_n = _batch_ce(batches[0], active=True)
    _b0_noop, _ = _batch_ce(batches[0], active=False)
    startup_loss_check(_b0_act / max(_b0_n, 1), _b0_noop / max(_b0_n, 1),
                       tag=log_tag, log_fn=log_fn)
    chunk_check: dict | None = None
    if chunked:
        # Wiring cross-check (chunked_ce module docstring) on a BOUNDED
        # slice: the shortest row of batch 0, cut to <= chunk_check_max_len
        # positions (never fewer than one target), scored by BOTH
        # implementations with the object active. The slice caps the
        # check's full-logits allocation below one chunk's transient, so
        # it is never the peak and cannot OOM where the loop fits — batch
        # 0 of a worst-case memory probe is ~8k tokens wide. bf16
        # kernel-tiling noise is <~1e-3 relative on a mean over hundreds
        # of tokens; a wrong body module or a dropped softcap is ~1e-1.
        # Refuse in between.
        _cb = _chunk_check_slice(batches, chunk_check_max_len)
        if (int(_cb.input_ids.shape[1]) > chunk_check_max_len
                or int((_cb.labels[:, 1:] != -100).sum()) < 1):
            raise RuntimeError(  # postcondition — the bound IS the point
                f"{log_tag}: cross-check slice violates its bound "
                f"({tuple(_cb.input_ids.shape)} positions, cap "
                f"{chunk_check_max_len})")
        _cb_chunk, _cb_n = _batch_ce(_cb, active=True)
        _cb_full, _ = _batch_ce(_cb, active=True, impl="full")
        ce_c = _cb_chunk / max(_cb_n, 1)
        ce_f = _cb_full / max(_cb_n, 1)
        rel = abs(ce_c - ce_f) / max(1.0, abs(ce_f))
        chunk_check = {"check_positions": int(_cb.input_ids.shape[1]),
                       "check_targets": int(_cb_n), "ce_full": ce_f,
                       "ce_chunked": ce_c, "rel_diff": rel,
                       "rtol": float(chunk_check_rtol),
                       "max_len": chunk_check_max_len}
        print(f"[{log_tag}] chunked_ce cross-check on one bounded row "
              f"[:{_cb.input_ids.shape[1]}] ({_cb_n} targets): "
              f"full={ce_f:.6f} chunked={ce_c:.6f} rel_diff={rel:.2e} "
              f"(rtol {chunk_check_rtol:g})", flush=True)
        if log_fn:
            log_fn({f"{log_tag}/chunk_check_rel_diff": rel}, 0)
        if not (rel <= float(chunk_check_rtol)):
            raise RuntimeError(
                f"{log_tag}: chunked_ce disagrees with the full-logits CE on "
                f"batch 0 (full {ce_f:.6f} vs chunked {ce_c:.6f}, rel "
                f"{rel:.3e} > {chunk_check_rtol:g}) — a wiring fault "
                "(wrong body module / missing softcap), not kernel noise; "
                "refusing to train.")

    total_steps = epochs * math.ceil(len(batches) / accum_steps)
    if warmup_steps is None:
        warmup_steps = min(500, max(1, total_steps // 10))

    # The schedule's lr for 0-based optimizer step ``i`` — lockstep's shared
    # factory (the psr mse loop's exact arithmetic), so the driver's
    # warmup_frac resolution (epochs x ceil(ceil(n/bs)/accum)) matches this
    # loop and 'warmup_cosine' means one thing across trainers. Imported
    # lazily (the bidpo convention: lockstep imports this module lazily too).
    from .lockstep import _lr_at_factory
    _lr_at = _lr_at_factory(lr, lr_schedule, warmup_steps, total_steps)

    stopper: EarlyStopper | None = (
        EarlyStopper(early_stop or EarlyStopSpec()) if h_batches else None)
    heldout_curve: list[float] = []
    heldout_noop_ce: float | None = None

    history: list[float] = []
    step = 0
    start_epoch = 0
    resume_order: list[int] | None = None
    resume_batches_done = 0
    ckpt_last_step = 0
    # The optimizer step a checkpoint restored this fit to (None = fresh):
    # the driver's proof that a continuation really resumed.
    resumed_from_step: int | None = None
    # Realized-batch accounting carried by the checkpoint: the
    # predecessor's latched first-epoch vector (or None), and the per-group
    # row counts of the epoch the checkpoint sits in, so a resume at an
    # epoch BOUNDARY — batches_done == n_batches, nothing left to run in
    # that epoch — still publishes the completed epoch's vector instead of
    # latching the empty one its own iteration realizes.
    resume_realized: list[int] | None = None
    resume_epoch_eff: list[int] = []
    # 1-based epochs whose end-of-epoch held-out value was REPLAYED under a
    # restarted lr and therefore not ruled on — carried by every
    # checkpoint so a successor's record is complete.
    replayed_partial_epochs: list[int] = []
    if checkpoint_every is not None:
        ck = load_checkpoint(
            checkpoint_dir, checkpoint_key,
            expect={"trainer": "train_object_ce",
                    "n_batches": len(batches)})
        if ck is not None:
            with torch.no_grad():
                for p, saved in zip(obj.trainable, ck["trainable"]):
                    p.data.copy_(saved.to(p.data.device))
            opt.load_state_dict(ck["opt_state"])
            rng.setstate(ck["rng_state"])
            torch.set_rng_state(ck["torch_rng_state"])
            restore_cuda_rng(ck.get("cuda_rng_states"))
            history = list(ck["history"])
            step = int(ck["step"])
            resumed_from_step = step
            start_epoch = int(ck["epoch"])
            resume_order = [int(b) for b in ck["order"]]
            resume_batches_done = int(ck["batches_done"])
            ckpt_last_step = step
            heldout_curve = [float(v) for v in ck.get("heldout_curve") or []]
            heldout_noop_ce = ck.get("heldout_noop_ce")
            replayed_partial_epochs = [
                int(e) for e in ck.get("replayed_partial_epochs") or []]
            resume_realized = ([int(v) for v in ck["realized_eff_batch"]]
                               if ck.get("realized_eff_batch") else None)
            resume_epoch_eff = [int(v) for v in ck.get("epoch_eff") or []]
            st = ck.get("stopper")
            if stopper is not None and st is not None:
                stopper.best = st["best"]
                stopper.best_epoch = st["best_epoch"]
                stopper.epochs_seen = st["epochs_seen"]
                stopper._since_best = st["since_best"]
                stopper._snapshot = st["snapshot"]
            print(f"[fitckpt] train_object_ce RESUMING from checkpoint: "
                  f"epoch {start_epoch}, micro-batch "
                  f"{resume_batches_done}/{len(batches)} in epoch, "
                  f"optimizer step {step}", flush=True)
    if replay_exempt_epoch is not None:
        # The exemption is a claim about THIS process's resume position:
        # refuse to run a fit that did not resume (a fresh start would
        # silence a real epoch's verdict), or one whose checkpoint sits
        # BEFORE the named epoch (no continuation moves backwards; a
        # successor's later checkpoint may sit past it — then nothing is
        # exempt here and the record rides in from the checkpoint).
        if resumed_from_step is None:
            raise RuntimeError(
                f"replay_exempt_epoch={replay_exempt_epoch} but no "
                "checkpoint was resumed — the exemption names the epoch a "
                "continuation resumed into; refusing to run a scratch fit "
                "under it")
        if start_epoch < replay_exempt_epoch:
            raise RuntimeError(
                f"replay_exempt_epoch={replay_exempt_epoch} but the resumed "
                f"checkpoint sits at epoch index {start_epoch} — a "
                "checkpoint that predates the replayed epoch cannot be the "
                "continuation's; refusing")
        if start_epoch == replay_exempt_epoch:
            print(f"[{log_tag}] R2-D77 replay_exempt: epoch "
                  f"{replay_exempt_epoch + 1} is the REPLAYED remainder of "
                  f"the continuation's checkpoint epoch (micro-batches "
                  f"{resume_batches_done + 1}..{len(batches)} under the "
                  "restarted lr) — its held-out value will be logged, not "
                  "ruled on; patience counts from the first FULL epoch",
                  flush=True)
        else:
            print(f"[{log_tag}] R2-D77 replay_exempt: resumed at epoch index "
                  f"{start_epoch}, past the replayed epoch "
                  f"{replay_exempt_epoch + 1} — no exemption in this "
                  f"process (record so far: {replayed_partial_epochs})",
                  flush=True)
    if h_batches and heldout_noop_ce is None:
        # The bare-base denominator (contract (ii)): computed ONCE before
        # epoch 1 with the object INACTIVE — nullcontext, the exact
        # unmodified base model — under the same micro-averaging as every
        # per-epoch read. Rides the checkpoint; a mid-fit resume restored
        # it above and skips this.
        heldout_noop_ce = _heldout_ce(active=False)
        if log_fn:
            log_fn({f"{log_tag}/heldout_noop_ce": heldout_noop_ce}, 0)
    if stopper is not None and not heldout_curve:
        # Pre-training baseline point (curve only — never a stopper
        # update, so the init can never become the restore target; the
        # psr/lockstep epoch -1 rule).
        heldout_curve.append(_heldout_ce(active=True))
        if log_fn:
            log_fn({f"{log_tag}/heldout_loss_init": heldout_curve[-1]}, 0)

    epochs_run = start_epoch
    stopped_early = False
    stopped_by_max_steps = False
    tokens_loss_total = tokens_real_total = tokens_padded_total = 0
    train_wall_s = 0.0
    # The fit's realized vector: the first epoch completed in this process
    # (a resumed process's is the resumed epoch's SUFFIX — realized_batch's
    # floor check), except when the checkpoint already carries one. A
    # predecessor that latched a vector hands it on; a predecessor that
    # checkpointed on the LAST group of an epoch (batches_done ==
    # n_batches: the process died in the held-out eval that followed) hands
    # on that epoch's complete accounting, because the resumed iteration of
    # it runs no group at all and — on the final epoch — neither does
    # anything after it (the empty vector used to be latched and
    # rejected by the validator; skipping empty epochs alone would leave a
    # final-epoch boundary resume with no vector).
    realized_eff_batch: list[int] | None = None
    if resumed_from_step is not None:
        if resume_realized:
            realized_eff_batch = list(resume_realized)
        elif resume_epoch_eff and resume_batches_done >= len(batches):
            realized_eff_batch = list(resume_epoch_eff)
    if device.type == "cuda":
        # Statistics only (docstring): the probe / bench read the LOOP's
        # peak after this call; the startup passes above are no-grad
        # transients below it. Synchronise first so nothing in flight
        # lands in the new window.
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs):
        if resume_order is not None and epoch == start_epoch:
            order = resume_order
            skip = resume_batches_done
        else:
            order = shuffled_epoch_order(rng, len(batches))
            skip = 0
        epoch_eff: list[int] = []
        # The per-group row counts THIS epoch realizes, from the shared
        # helper the validator reproduces; appended one at a time as each
        # group completes, so a bounded / diverged run leaves exactly the
        # partial vector it actually ran.
        epoch_group_rows = group_row_counts(n_rows, order, accum_steps, skip)
        # Accumulation groups along the shuffled order; groups NEVER span
        # epochs — the ragged final group is its own smaller step
        # (contract (i)).
        for gi, g0 in enumerate(range(skip, len(order), accum_steps)):
            group = order[g0:g0 + accum_steps]
            grp_tok = sum(n_tok[b] for b in group)
            grp_rows = sum(n_rows[b] for b in group)
            acc_loss = acc_ce = acc_reg = 0.0
            t_group0 = time.perf_counter()
            # ONE zero_grad per optimizer step, at the group's START (it
            # used to sit in the first micro-batch's slot — equivalent
            # there, since the forward never touches .grad — but a rank
            # whose strided share of a ragged tail group is EMPTY must still
            # clear last step's gradient before the all-reduce below).
            opt.zero_grad()
            # This rank's share of the group: the strided micro-batches
            # ``group[rank::world]`` in the group's own order — the whole
            # group at width 1. The token-share weights below stay GLOBAL
            # (over the whole group), so the per-rank partial sums add up
            # to exactly the single-process accumulated step.
            for bi in group[rank::world]:
                sb = batches[bi]
                # Token-share weight: this micro-batch's CE-target share
                # of the GROUP (its own loss denominator), so the summed
                # gradient equals one true concatenated batch's. A
                # single-batch group's share is exactly 1.0 (x/x, and a
                # multiply by 1.0 is exact in IEEE-754), so the
                # accum_steps=1 step stays byte-identical.
                w_ce = n_tok[bi] / grp_tok
                w_reg = n_rows[bi] / grp_rows
                reg = None
                # Forward + loss assembly + BACKWARD all inside apply()
                # (contract (iii)): the checkpointing recompute during
                # backward must see the hooks live.
                with obj.apply(model, scale=1.0):
                    if chunked:
                        # chunked_ce module docstring, steps 1-4. The
                        # per-chunk grad scale is d(step_loss)/d(nll_sum)
                        # = w_ce / n_tok — exactly the full path's
                        # w_ce * mean(...) chain — so the joint backward
                        # below reproduces step_loss.backward().
                        h = forward_last_hidden(model, sb.input_ids,
                                                sb.attention_mask)
                        h_sel, targets = completion_hidden(h, sb)
                        nll_sum, g_sel = chunked_ce(
                            _lm_head, _softcap, h_sel, targets,
                            chunk_tokens=chunk_tokens,
                            grad_scale=w_ce / n_tok[bi])
                        ce = nll_sum / n_tok[bi]  # detached fp32 scalar
                        step_loss = w_ce * ce
                        if reg_coeff:
                            reg = obj.gate_reg_term(sb.attention_mask, None)
                            reg_scaled = reg_coeff * (w_reg * reg)
                            step_loss = step_loss + reg_scaled.detach()
                            # ONE body backward for CE + regulariser
                            # jointly (never two passes): dL/dh_sel is the
                            # assembled chunk gradient, the scalar reg gets
                            # the implicit ones.
                            torch.autograd.backward([h_sel, reg_scaled],
                                                    [g_sel, None])
                        else:
                            torch.autograd.backward([h_sel], [g_sel])
                    else:
                        logits = model(input_ids=sb.input_ids,
                                       attention_mask=sb.attention_mask).logits
                        s_logits = _gather_completion_logits(logits, sb)
                        labels = sb.labels[:, 1:]
                        targets = labels[labels != -100]
                        ce = F.cross_entropy(s_logits.float(), targets)
                        step_loss = w_ce * ce
                        if reg_coeff:
                            # Inside the ctx (the gate stash clears at
                            # exit), via the object's ONE shared assembly.
                            reg = obj.gate_reg_term(sb.attention_mask, None)
                            step_loss = step_loss + reg_coeff * (w_reg * reg)
                        step_loss.backward()
                acc_loss += float(step_loss.detach())
                acc_ce += float(ce.detach()) * w_ce
                if reg is not None:
                    acc_reg += float(reg.detach()) * w_reg
            # Data-parallel: SUM the per-rank partial gradients — the
            # object's trainable (adapter) parameters only, one flat
            # all-reduce; a rank that forwarded nothing contributes zeros —
            # and the fp64 loss shares. After this every rank holds the
            # identical global gradient and loss. Identity at width 1.
            _ddp.allreduce_grads_and_stats(obj.trainable, [], par)
            acc_loss, acc_ce, acc_reg = _ddp.allreduce_vec(
                [acc_loss, acc_ce, acc_reg], par)
            # Group end: ONE clip, on the ACCUMULATED gradient.
            grad_norm_t = None
            if math.isfinite(max_grad_norm):
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    obj.trainable, max_grad_norm)
            cur_lr = _lr_at(step)
            for grp in opt.param_groups:
                grp["lr"] = cur_lr
            opt.step()
            step += 1
            # The float() syncs above already drained the stream, so this
            # wall time is the optimizer step's (forward+backward+step),
            # not a queue depth. Token counts are the pre-computed
            # per-batch sums — no device sync of their own.
            dt_step = time.perf_counter() - t_group0
            grp_real = sum(n_real[b] for b in group)
            grp_padded = sum(n_padded[b] for b in group)
            tokens_loss_total += grp_tok
            tokens_real_total += grp_real
            tokens_padded_total += grp_padded
            train_wall_s += dt_step
            history.append(acc_loss)
            epoch_eff.append(epoch_group_rows[gi])   # == grp_rows
            if not math.isfinite(acc_loss):
                raise TrainingDiverged(
                    f"{log_tag}: non-finite training loss ({acc_loss!r}) "
                    f"at optimizer step {step} — never train through NaN.")
            if log_fn:
                rec = {f"{log_tag}/ce_loss": acc_loss,
                       f"{log_tag}/ce": acc_ce,
                       f"{log_tag}/lr": cur_lr,
                       # Throughput record (gpu-util skill): loss targets,
                       # real encoded tokens, padded width, wall seconds
                       # for THIS optimizer step.
                       f"{log_tag}/loss_tokens": grp_tok,
                       f"{log_tag}/real_tokens": grp_real,
                       f"{log_tag}/padded_tokens": grp_padded,
                       f"{log_tag}/sec_per_step": dt_step,
                       "epoch": epoch}
                if reg_coeff:
                    rec[f"{log_tag}/gate_reg"] = acc_reg
                if grad_norm_t is not None:
                    rec[f"{log_tag}/grad_norm"] = float(grad_norm_t)
                log_fn(rec, step)
            if (checkpoint_every is not None
                    and step - ckpt_last_step >= checkpoint_every):
                # Group end == optimizer-step boundary (contract (iv)):
                # no partial accumulation state exists here.
                stopper_state = None
                if stopper is not None:
                    stopper_state = {
                        "best": stopper.best,
                        "best_epoch": stopper.best_epoch,
                        "epochs_seen": stopper.epochs_seen,
                        "since_best": stopper._since_best,
                        "snapshot": (None if stopper._snapshot is None else
                                     [t.detach().cpu().clone()
                                      for t in stopper._snapshot]),
                    }
                # Data-parallel: rank 0 writes (the replicas are bit-equal
                # — the assert before the return — so its state IS the
                # run's); every rank keeps the cadence bookkeeping and
                # every rank RESUMES the same file at start.
                if is_main:
                    save_checkpoint(checkpoint_dir, {
                        "checkpoint_key": checkpoint_key,
                        "trainer": "train_object_ce",
                        "n_batches": len(batches),
                        "epoch": epoch,
                        "batches_done": g0 + len(group),
                        "order": list(order),
                        "rng_state": rng.getstate(),
                        "step": step,
                        "history": list(history),
                        "trainable": [p.detach().cpu().clone()
                                      for p in obj.trainable],
                        "opt_state": to_cpu(opt.state_dict()),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_states": cuda_rng_states(),
                        "heldout_curve": list(heldout_curve),
                        "heldout_noop_ce": heldout_noop_ce,
                        "stopper": stopper_state,
                        "replayed_partial_epochs":
                            list(replayed_partial_epochs),
                        # realized-batch accounting: the latched
                        # vector and this epoch's groups so far — complete
                        # when the checkpoint sits on the epoch's last group
                        "realized_eff_batch": (
                            None if realized_eff_batch is None
                            else list(realized_eff_batch)),
                        "epoch_eff": list(epoch_eff),
                        "fit_width": world,
                    })
                    print(f"[fitckpt] train_object_ce checkpoint written: "
                          f"epoch {epoch}, micro-batch {g0 + len(group)}/"
                          f"{len(batches)} in epoch, optimizer step {step}"
                          + (f" (rank 0 of {world})" if world > 1 else ""),
                          flush=True)
                ckpt_last_step = step
            if max_steps is not None and step >= max_steps:
                stopped_by_max_steps = True
                break
        if realized_eff_batch is None and epoch_eff:
            # An epoch that ran no group (a boundary resume whose checkpoint
            # predates the carried accounting) latches nothing, so the next
            # completed epoch is recorded instead of an empty vector.
            realized_eff_batch = epoch_eff
        if stopped_by_max_steps:
            # A bounded run: no epoch completed, no held-out ruling, no
            # stopper (refused with max_steps above). epochs_run stays at
            # the completed count.
            print(f"[{log_tag}] max_steps={max_steps} reached at optimizer "
                  f"step {step} — stopping the bounded run.", flush=True)
            break
        epochs_run = epoch + 1
        if stopper is not None:
            hl = _heldout_ce(active=True)
            heldout_curve.append(hl)
            # The replayed remainder of the continuation's checkpoint
            # epoch: logged and kept in the curve, flagged, but
            # never a strike nor a new best. Reached only when this
            # process resumed INTO that epoch (start_epoch == the index;
            # checked above), so a successor past it never exempts.
            exempt = (replay_exempt_epoch is not None
                      and epoch == replay_exempt_epoch)
            if log_fn:
                rec = {f"{log_tag}/heldout_loss": hl, "epoch": epoch}
                if replay_exempt_epoch is not None:
                    rec[f"{log_tag}/replayed_partial"] = int(exempt)
                log_fn(rec, step)
            if exempt:
                replayed_partial_epochs.append(epoch + 1)
                stopper.observe_exempt()
                print(f"[{log_tag}] epoch {epoch + 1} held-out CE {hl:.6f} "
                      f"(REPLAYED partial epoch — not ruled on; incumbent "
                      f"best {stopper.best:.6f} @ epoch {stopper.best_epoch}"
                      f", since_best {stopper._since_best} unchanged)",
                      flush=True)
            elif stopper.update(hl, obj.trainable):
                stopped_early = True
                if log_fn:
                    log_fn({f"{log_tag}/stopped_epoch": epoch + 1,
                            f"{log_tag}/best_epoch": stopper.best_epoch},
                           step)
        if epoch_cb is not None:
            # EPOCH-BOUNDARY hook: after this epoch's held-out pass and the stopper's ruling, with the
            # object as it stands at the end of the epoch (a stopping epoch's weights are NOT yet restored
            # to the best epoch — the restore runs after the loop). The driver saves a DURABLE copy (epoch
            # 1, epoch 2 and the final epoch are the named artefacts; never rotated by the step-interval
            # checkpoint above). Every epoch, stopping ones included; the callback decides what to keep. A
            # callback error is the caller's.
            epoch_cb(epoch, step, obj, list(heldout_curve), bool(stopped_early))
        if stopped_early:
            break
    if stopper is not None:
        # Best-epoch restore, at the cap too (the restore-at-cap rule);
        # everything downstream (the PSR dead-gate guard, the CE floor
        # ratio) rules on the weights that would actually ship.
        stopper.restore_best(obj.trainable)
    # Data-parallel: the replicas must be bit-equal here (identical
    # all-reduced gradients -> identical AdamW steps); a divergence is a
    # LOUD error, never a silently wrong artifact. No-op at width 1.
    _ddp.assert_replicas_equal(log_tag, list(obj.trainable), par)
    if diagnostics is not None:
        best_ce = (stopper.best if stopper is not None
                   and math.isfinite(stopper.best) else None)
        diagnostics.update({
            # The data-parallel width this fit REALLY ran at (1 = the
            # single-process path); the driver records it on the sidecar
            # and fit_selection.json.
            "fit_width": world,
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "heldout_curve": list(heldout_curve),
            "best_epoch": stopper.best_epoch if stopper else None,
            # Both spellings, deliberately: best_heldout_loss is the
            # driver/report standard shape; best_heldout_ce +
            # heldout_noop_ce are the CE floor guard's numerator and
            # bare-base denominator.
            "best_heldout_loss": best_ce,
            "best_heldout_ce": best_ce,
            "heldout_noop_ce": heldout_noop_ce,
            "realized_eff_batch": list(realized_eff_batch or []),
            "steps_per_epoch": math.ceil(len(batches) / accum_steps),
            "optimizer_steps_run": step,
            "resumed_from_step": resumed_from_step,
            # Which 1-based epochs of the curve were replayed partial
            # epochs (logged, never ruled on) — the whole record,
            # including epochs a predecessor process exempted.
            "replayed_partial_epochs": list(replayed_partial_epochs),
            "replay_exempt_epoch": replay_exempt_epoch,
            # The loss implementation is a fit-identity dimension
            # (recorded here for the sidecar; the driver fingerprints it),
            # plus the throughput totals for THIS process's steps (a resume
            # counts from its restart) and the bounded-run flag.
            "loss_impl": loss_impl,
            "chunk_tokens": chunk_tokens if chunked else None,
            "chunk_check": chunk_check,
            "stopped_by_max_steps": stopped_by_max_steps,
            "tokens_loss_total": tokens_loss_total,
            "tokens_real_total": tokens_real_total,
            "tokens_padded_total": tokens_padded_total,
            "train_wall_s": train_wall_s,
        })
    return history


def ce_loss_impl_parity(model, tok, obj, examples: list[DistillExample], *,
                        max_len: int, reg_coeff: float = 0.0,
                        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                        perturb: float = 0.0) -> dict:
    """Diagnostic: ONE micro-batch's step loss and dL/dparams under
    ``loss_impl="full"`` vs ``"chunked_ce"``, on the live model in its live
    dtype — the wiring CHECK at production scale. The CPU tests prove fp32
    exactness; this MEASURES the bf16 gap and decomposes it. A dropped
    softcap or a skipped hook shows up as a CE mismatch (the two CE values
    are compared directly) and as a head-side gap far above the paired
    head measurement below. ``perturb`` > 0 nudges the object's
    parameters by that much (deterministic, seed 0) for the measurement so
    a zero-initialised object still has non-trivial gradients on every
    parameter group; the parameters and ``.grad`` are RESTORED after.
    Returns the two losses, their relative gap, the relative L2 gap and
    cosine of the concatenated gradients, plus a DECOMPOSITION of that
    gap. The two paths share the body forward (the same ``model(...)``
    call; ``logits_to_keep`` only changes the wrapper's head call), so
    they differ ONLY at the head side — the paired measurement:

    - ``head_paired_*``: from ONE captured hidden state, dL/dh at the
      completion positions via the full-logits route (head over every
      position, softcap, gather, mean CE) vs the chunked route (gather,
      head per chunk, softcap, summed CE). Same realization, no body
      involvement — the actual numerical difference the two paths feed
      into an otherwise identical body backward (which then propagates,
      and may amplify or cancel, that difference).

    Three further comparisons with the body batching unchanged:

    - ``repeat_full_*``: the full path run twice — nondeterministic-kernel
      repeat noise (0 on deterministic kernels; reported, not assumed);
    - ``chunk_split_*``: gather-first CE as ONE all-targets chunk vs the
      production ``chunk_tokens`` — the chunk SPLIT alone (head matmul M
      dimension + fp32 summation order);
    - ``single_chunk_vs_full_*``: the one-chunk gather-first path vs the
      full path — gathering BEFORE the head (M = N targets) vs after
      (M = B x S incl. pads/prompt), softcap on either.

    ``rowwise_sensitivity_*`` (the full path with the rows batched vs one
    at a time, CE only) is a broader, SEPARATE sensitivity comparison — it
    changes every body kernel's batch shape — reported for context with
    NO bound semantics: a wider perturbation does not imply a larger
    error. Exactness is proven in fp32 by the CPU tests;
    none of these legs is a proof about bf16, only measurements."""
    device = next(model.parameters()).device
    for p in model.parameters():
        p.requires_grad_(False)
    for p in obj.trainable:
        p.data = p.data.to(device)
    enc = [_encode_aligned(tok, ex, max_len)[1] for ex in examples]
    batch = collate(enc, tok.pad_token_id).to(device)
    n_tok = int((batch.labels[:, 1:] != -100).sum())
    saved = [p.detach().clone() for p in obj.trainable]
    if perturb:
        g = torch.Generator(device="cpu").manual_seed(0)
        with torch.no_grad():
            for p in obj.trainable:
                noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
                p.add_(noise.to(p.device, p.dtype) * perturb)

    def _grads(impl: str, chunk: int | None = None
               ) -> tuple[float, list[torch.Tensor]]:
        chunk = int(chunk_tokens) if chunk is None else int(chunk)
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
                val = float(loss.detach())
            else:
                h = forward_last_hidden(model, batch.input_ids,
                                        batch.attention_mask)
                h_sel, targets = completion_hidden(h, batch)
                nll, g_sel = chunked_ce(lm_head_of(model),
                                        final_logit_softcap(model), h_sel,
                                        targets, chunk_tokens=chunk,
                                        grad_scale=1.0 / n_tok)
                val = float(nll / n_tok)
                if reg_coeff:
                    reg = reg_coeff * obj.gate_reg_term(batch.attention_mask,
                                                        None)
                    torch.autograd.backward([h_sel, reg], [g_sel, None])
                    val += float(reg.detach())
                else:
                    torch.autograd.backward([h_sel], [g_sel])
        grads = [(p.grad.detach().float().clone() if p.grad is not None
                  else torch.zeros_like(p, dtype=torch.float32))
                 for p in obj.trainable]
        return val, grads

    def _grads_full_ce_only(rowwise: bool) -> list[torch.Tensor]:
        """CE-only "full"-path gradients, the rows batched (one padded
        forward) or one at a time (accumulated) — the noise-floor pair."""
        for p in obj.trainable:
            p.grad = None
        groups = ([[e] for e in enc] if rowwise else [enc])
        for grp in groups:
            b = collate(grp, tok.pad_token_id).to(device)
            with obj.apply(model, scale=1.0):
                logits = model(input_ids=b.input_ids,
                               attention_mask=b.attention_mask).logits
                s_logits = _gather_completion_logits(logits, b)
                labels = b.labels[:, 1:]
                targets = labels[labels != -100]
                # sum / N over the WHOLE row set: the same scalar objective
                # whichever way the rows are grouped
                (F.cross_entropy(s_logits.float(), targets,
                                 reduction="sum") / n_tok).backward()
        return [(p.grad.detach().float().clone() if p.grad is not None
                 else torch.zeros_like(p, dtype=torch.float32))
                for p in obj.trainable]

    def _gap(a: list[torch.Tensor], b: list[torch.Tensor]) -> tuple:
        va = torch.cat([g.reshape(-1) for g in a])
        vb = torch.cat([g.reshape(-1) for g in b])
        den = float(va.norm())
        num = float((va - vb).norm())
        cos = (float((va * vb).sum()) / max(den * float(vb.norm()), 1e-30)
               if den > 0 else float("nan"))
        return den, num / max(den, 1e-30), cos

    def _head_paired() -> dict:
        """dL/dh at the completion positions from ONE captured hidden
        state, full-logits route vs chunked route (both for L = mean CE
        over the N targets; CE only; the body excluded by detaching h)."""
        head, cap = lm_head_of(model), final_logit_softcap(model)
        with obj.apply(model, scale=1.0), torch.no_grad():
            h = forward_last_hidden(model, batch.input_ids,
                                    batch.attention_mask)
        h = h.detach()
        # full route: head over EVERY position, softcap, gather, mean CE
        hf = h.clone().requires_grad_(True)
        logits = head(hf)
        if cap is not None:
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        s_logits = _gather_completion_logits(logits, batch)
        labels = batch.labels[:, 1:]
        targets = labels[labels != -100]
        loss_f = F.cross_entropy(s_logits.float(), targets)
        (gh,) = torch.autograd.grad(loss_f, hf)
        g_full_sel = gh[:, :-1][labels != -100].float()
        # chunked route on the same h
        h_sel, targets_c = completion_hidden(h, batch)
        nll_c, g_chunk = chunked_ce(head, cap, h_sel, targets_c,
                                    chunk_tokens=int(chunk_tokens),
                                    grad_scale=1.0 / n_tok)
        den, rel, cos = _gap([g_full_sel], [g_chunk.float()])
        ce_f, ce_c = float(loss_f.detach()), float(nll_c / n_tok)
        return {"head_paired_ce_rel_diff": abs(ce_f - ce_c) / max(1.0, abs(ce_f)),
                "head_paired_grad_l2_full": den,
                "head_paired_grad_rel_l2_diff": rel,
                "head_paired_grad_cosine": cos}

    try:
        ce_f, g_f = _grads("full")
        _ce_f2, g_f2 = _grads("full")                      # repeat noise
        ce_c, g_c = _grads("chunked_ce")                   # production chunk
        _ce_c1, g_c1 = _grads("chunked_ce", chunk=max(n_tok, 1))  # 1 chunk
        rw_batched = _grads_full_ce_only(rowwise=False)
        rw_rowwise = _grads_full_ce_only(rowwise=True)
        head = _head_paired()
    finally:
        with torch.no_grad():
            for p, s0 in zip(obj.trainable, saved):
                p.data.copy_(s0)
                p.grad = None
    den, rel, cos = _gap(g_f, g_c)
    _d, rep_rel, rep_cos = _gap(g_f, g_f2)
    _d, split_rel, split_cos = _gap(g_c1, g_c)
    _d, one_rel, one_cos = _gap(g_f, g_c1)
    _d, rw_rel, rw_cos = _gap(rw_batched, rw_rowwise)
    return {"rows": len(enc), "n_targets": n_tok, "dtype": "torch.float32",
            "model_dtype": str(next(model.parameters()).dtype),
            "ce_full": ce_f, "ce_chunked": ce_c,
            "ce_rel_diff": abs(ce_f - ce_c) / max(1.0, abs(ce_f)),
            "grad_l2_full": den, "grad_rel_l2_diff": rel,
            "grad_cosine": cos,
            # PAIRED head-side difference on one captured hidden state
            **head,
            # comparisons with the body batching UNCHANGED
            "repeat_full_grad_rel_l2_diff": rep_rel,
            "repeat_full_grad_cosine": rep_cos,
            "chunk_split_grad_rel_l2_diff": split_rel,
            "chunk_split_grad_cosine": split_cos,
            "single_chunk_vs_full_grad_rel_l2_diff": one_rel,
            "single_chunk_vs_full_grad_cosine": one_cos,
            # a broader, SEPARATE sensitivity comparison (body batching
            # CHANGED; CE only) — context, no bound semantics
            "rowwise_sensitivity_grad_rel_l2_diff": rw_rel,
            "rowwise_sensitivity_grad_cosine": rw_cos,
            "calibration_note": (
                "head_paired: dL/dh at the completion positions from ONE "
                "captured hidden state, full-logits head route vs chunked "
                "route — the paired, same-realization difference at the "
                "only place the two paths' computations differ (the body "
                "forward is the same call; the body backward then "
                "propagates, and may amplify or cancel, that difference). "
                "repeat_full: same computation twice (kernel repeat noise); "
                "chunk_split: one all-targets chunk vs the production chunk "
                "(head M dim + fp32 sum order only); single_chunk_vs_full: "
                "gather-before vs gather-after the head (head shape only); "
                "rowwise_sensitivity: full path rows batched vs row-wise, "
                "CE only — a broader, separate comparison (every body "
                "kernel's batch shape changes) with no bound semantics: a "
                "wider perturbation does not imply a larger error. None of "
                "these is a proof about bf16; exactness is proven in fp32 "
                "by tests/test_chunked_ce.py."),
            "perturb": perturb, "chunk_tokens": int(chunk_tokens)}
