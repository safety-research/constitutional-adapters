"""Lockstep teacher sharing (efficiency optimisation 4).

In the distillation loss the teacher is the SI-prompted frozen base with the
intervention OBJECT INACTIVE, so for a given (SI, example) the teacher forward
is BIT-IDENTICAL for every arm — it does not depend on the object at all. The
single-cell path recomputes it once per arm, i.e. ~10x per SI. Lockstep trains
several compatible arms in ONE process where each teacher batch is computed
ONCE and its (gathered, completion-position) logits are handed to every arm's
student step before moving to the next batch. Nothing is cached to disk (the
full teacher logits would be 492 x 512 x 128k ~ 64 GB — lockstep exists exactly
to avoid that): each batch's teacher logits live only for the span of that
batch's N student steps, then are freed.

Numerical contract (non-negotiable): a lockstep-trained object is BITWISE
identical to the same arm trained solo, given the same seed, data order and
batch composition. This holds because the ONLY thing lockstep changes about
an arm's trajectory is WHERE the teacher logits come from, and the shared
forward computes exactly the tensor the solo trainer's own teacher forward
would (same frozen model, same teacher batch, same ``no_grad`` context, same
``_gather_completion_logits``). Everything else — each arm's object, its
optimiser and state, its RNG stream, its per-step math — is untouched and
independent, so adding an arm to a group cannot perturb another arm.
``tests/test_lockstep.py`` proves the equality against the REAL registry
trainers (``methods.ARMS[arm].train``), not a reimplementation.

Compatibility (``lockstep_compatible``): an arm may join a teacher-sharing
group iff its training consumes the teacher's LOGIT DISTRIBUTION through the
shared soft divergence — forward KL / reverse KL / JS — i.e. it routes through
``train_object_kl(loss='soft')`` or ``train_steered_lm(loss in SOFT_LOSSES)``.
Arms that consume something else are EXCLUDED (they train solo), never coerced:

  - psr (mse), psr_mse_all      -> teacher ACTIVATIONS (layerwise MSE), no logits
  - loreft, reft_r1, lofit      -> cross-entropy on teacher TOKENS, no teacher fwd
  - psr_kl_hard, psr_kl_all_hard-> ``hard`` loss: teacher TOKENS, not a distribution
  - bidpo                       -> preference pairs, no per-example teacher logits
  - antidote, alphasteer,
    alphakl_null/soft/fisher    -> closed-form / probe fits, not the KL loop
  - diffmean, caa_static_r      -> closed-form diff-in-means, not the KL loop

The per-step math is a faithful transcription of ``train_object_kl(loss=
'soft')`` (and, for the additive-vector arms, ``train_steered_lm``): both build
the SAME batches (``_encode_aligned`` after ``obj.inject`` on the student side,
teacher side un-injected), both step ``loss_scale * soft_divergence(loss,
t_logp, s_logp)`` with per-arm temperature, both use ``AdamW`` at the arm's lr.
For ``loss='soft'`` ``soft_divergence`` reduces to
``F.kl_div(s_logp, t_logp, log_target=True, reduction='batchmean')`` — exactly
``train_object_kl``'s objective. The PSR KL arms additionally carry the
anti-dead-gate regularizer (``LockstepArmParams.reg_coeff``, default 0.1 for
them, mirroring ``train_psr``): both paths add ``reg_coeff *
obj.gate_reg_term(...)`` — the object's ONE shared assembly — after the
divergence, so the bitwise contract extends to the regularized arms.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..models import num_layers
from .distill import (
    DEFAULT_PROBE_N,
    DistillExample,
    EarlyStopSpec,
    EarlyStopper,
    ProbeEstimate,
    _encode_aligned,
    _gather_completion_logits,
    _mean_sd_sem,
    _ratio_sem,
    loss_gather_masks,
    per_example_means,
    probe_family_table,
    probe_values_record,
    soft_divergence,
    soft_divergence_per_token,
    soft_divergence_topk,
    startup_loss_check,
    TokenizeProgress,
    steer_prompt_boundaries,
    SOFT_LOSSES,
    preflight_max_len,
)
from .fitckpt import (
    cuda_rng_states,
    load_checkpoint,
    resolve_checkpoint_kwargs,
    restore_cuda_rng,
    save_checkpoint,
    to_cpu,
)
from .prefix_cache import prefix_cache_for_batches
from .subsample import sha_hex, sha_rank_indices, stratified_sha_rank_indices
from .tokcache import cached_ids
from .traindata import Batch, collate

# Arms whose registry trainer consumes the teacher LOGIT DISTRIBUTION via the
# shared soft divergence (train_object_kl(loss='soft') or train_steered_lm with
# loss in SOFT_LOSSES). Membership is the lockstep-compatibility predicate.
# Kept as an explicit, documented set rather than sniffed from ARMS because the
# ArmSpec does not expose the trainer's loss; the module docstring records why
# each EXCLUDED arm is excluded, and test_lockstep pins this set against the
# registry so a new arm cannot silently change compatibility.
_LOCKSTEP_COMPATIBLE: frozenset[str] = frozenset({
    "vector",       # train_steered_lm(loss='soft')  — the distill_soft cell
    "vector_rkl",   # train_steered_lm(loss='rkl')   — divergence ablation
    "vector_js",    # train_steered_lm(loss='js')    — divergence ablation
    "vector_all",   # train_object_kl(loss='soft')
    "lora_r1",      # train_object_kl(loss='soft')
    "lora_r1_all",  # train_object_kl(loss='soft')
    "lora_r4",      # train_object_kl(loss='soft') — lora_r1_all's post-re-tune name
    "token_bos",    # train_object_kl(loss='soft')
    "token_faithful",  # train_object_kl(loss='soft', temperature=10, loss_scale=100)
    "psr_kl",       # train_psr(loss='kl') -> train_object_kl(loss='soft')
    "psr_kl_all",   # train_psr(loss='kl') -> train_object_kl(loss='soft')
})


def lockstep_compatible(arm: str) -> bool:
    """Whether ``arm`` may join a lockstep teacher-sharing group.

    True iff the arm's training consumes the SI-prompted teacher's logit
    DISTRIBUTION via the shared soft divergence (see the module docstring for
    the per-arm reasoning behind every exclusion). Callers partition their arm
    list with this predicate and train the incompatible remainder SOLO through
    the ordinary registry path — never coerced into a group."""
    return arm in _LOCKSTEP_COMPATIBLE


def lockstep_incompatible_reason(arm: str) -> str | None:
    """A short, human-readable reason ``arm`` cannot share a teacher forward,
    or ``None`` if it can. For logging when an arm falls back to solo."""
    if lockstep_compatible(arm):
        return None
    reasons = {
        "psr": "mse loss matches teacher ACTIVATIONS, not logits",
        "psr_mse_all": "mse loss matches teacher ACTIVATIONS, not logits",
        "psr_kl_hard": "hard loss consumes teacher TOKENS, not a distribution",
        "psr_kl_all_hard": "hard loss consumes teacher TOKENS, not a distribution",
        "loreft": "cross-entropy on teacher tokens; no teacher forward at all",
        "reft_r1": "cross-entropy on teacher tokens; no teacher forward at all",
        "lofit": "cross-entropy on teacher tokens; no teacher forward at all",
        "psr_ce_all": "cross-entropy on document tokens; no teacher forward at all",
        "lora_r4_ce": "cross-entropy on document tokens; no teacher forward at all",
        "bidpo": "preference pairs; no per-example teacher logits",
        "antidote": "closed-form PCA fit; not the KL training loop",
        "alphasteer": "closed-form null-space ridge fit; not the KL loop",
        "alphakl_null": "probe-ladder fit (ridge solve); not the KL loop",
        "alphakl_soft": "probe-ladder fit (ridge solve); not the KL loop",
        "alphakl_fisher": "probe-ladder fit (Fisher/JVP); not the KL loop",
        "diffmean": "closed-form diff-in-means extraction; not the KL loop",
        "caa_static_r": ("closed-form diff-in-means extraction (Class B, "
                         "fit-once, SI-free); not the KL loop"),
        # the sdf-base-steering DERIVED arms: never trained here at all,
        # so there is no training loop of any kind to share a teacher with
        "psr_ce_all_minus_ctl_distilled": (
            "a DERIVED object composed from two fitted parents "
            "(scripts/sdfbase/psr_residual_distill.py); never trained "
            "in-repo, so there is no teacher forward to share"),
        "lora_r4_ce_minus_ctl": (
            "a DERIVED object composed from two fitted parents "
            "(scripts/sdfbase/lora_subtract.py); never trained in-repo, so "
            "there is no teacher forward to share"),
    }
    return reasons.get(arm, "arm does not consume teacher logit distributions")


@dataclass(frozen=True)
class LockstepArmParams:
    """The soft-KL hyper-params an arm's registry trainer would use, so the
    lockstep step reproduces it exactly. ``lr``/``loss``, the distillation
    ``temperature``/``loss_scale`` and ``reg_coeff`` are the only per-arm
    knobs the shared soft-divergence step reads; everything else (epochs,
    batch_size, max_len, seed) is shared across the group (it must be, for
    the teacher batches to align).

    ``reg_coeff`` > 0 adds the PSR anti-dead-gate regularizer to the arm's
    step loss — ``reg_coeff * obj.gate_reg_term(...)``, the exact term
    train_psr's KL path adds through train_object_kl (both objectives
    carry it since the 2026-08-20 confound fix). Only meaningful for an
    object exposing ``gate_reg_term`` (PSRObject); the step refuses loudly
    otherwise.

    ``lr_schedule`` / ``warmup_steps`` / ``max_grad_norm`` are the PER-ARM
    optimiser-schedule opt-in (2026-08-24), mirroring train_psr's mse-loop
    treatment (same names, same warmup ramp ``lr*(i+1)/warmup``, same
    cosine decay to 0, same ``min(500, total//10)`` warmup default, same
    ``math.inf``-disables-clipping convention) so the vocabulary is one
    across trainers. THE DEFAULTS ARE TODAY'S BEHAVIOUR — ``None`` means
    no schedule, no warmup, no clipping, and the step loop touches
    nothing, so every arm that does not opt in stays bit-identical to its
    solo registry trainer (unlike train_psr's mse path, where ``None``
    resolves to ``warmup_cosine``: there the schedule IS the trainer
    default; here bit-compatibility with the shared KL loop is the
    default). An arm that DOES opt in deliberately departs from its solo
    trainer's fixed-lr loop for its own trajectory only — per-arm
    optimiser and RNG stream mean a scheduled arm cannot perturb its
    group partners (pinned by test_lockstep)."""

    loss: str = "soft"
    temperature: float = 1.0
    loss_scale: float = 1.0
    lr: float = 5e-3
    #: AdamW weight decay — exposed 2026-08-27 (re-tune): the optimizer had
    #: silently been applying AdamW's own default to every arm with nothing
    #: tuning or recording it. The default IS that 0.01, so an arm that does
    #: not set it stays byte-identical to its solo registry trainer (whose
    #: trainers carry the same exposed default).
    weight_decay: float = 0.01
    reg_coeff: float = 0.0
    lr_schedule: str | None = None      # None | "warmup_cosine" | "constant"
    warmup_steps: int | None = None     # None = min(500, total//10) when scheduled
    #: Gradient accumulation (twosi-model-sweeps re-tune@32): each optimizer
    #: step spans ``accum_steps`` consecutive micro-batches of the shared
    #: ``batch_size``, so effective batch = batch_size x accum_steps. The
    #: accumulated gradient EQUALS the gradient of one true batch of those
    #: rows: each micro-batch's divergence is weighted by its COMPLETION-
    #: TOKEN share of the group (batchmean's own denominator) and the
    #: gate regularizer by its ROW share (gate_regularizer's denominator)
    #: — bidpo's row-weighted contract, per-denominator; a flat
    #: 1/accum_steps would silently bias both whenever the group is ragged.
    #: Gradient clipping fires ONCE per optimizer step, on the ACCUMULATED
    #: gradient — never per micro-batch, which would silently reproduce
    #: micro-batch-level clipping inside an effective-32 run. 1 (the
    #: default) is byte-identical to the historical per-batch loop. Like
    #: the schedule keys, an accumulating arm deliberately departs from its
    #: solo registry trainer; per-arm optimiser/RNG isolation keeps it from
    #: perturbing group partners.
    accum_steps: int = 1
    #: Warmup as a FRACTION of this fit's own total optimizer steps
    #: (twosi-model-sweeps: HPO winners are recorded as warmup_frac so the
    #: production fit re-derives steps from its own step budget — never a
    #: copied step count from a subsampled trial). Mutually exclusive with
    #: ``warmup_steps``; resolved inside train_lockstep where total steps
    #: are known; requires an ``lr_schedule`` like warmup_steps does.
    warmup_frac: float | None = None
    max_grad_norm: float | None = None  # None = no clipping (inf also disables)


def default_lockstep_params(arm: str) -> LockstepArmParams:
    """The per-arm soft-KL params matching that arm's registry trainer.

    Mirrors ``methods.py``'s trainer wiring for the compatible arms: the
    divergence-ablation vectors carry their own divergence; the faithful soft
    token carries Radevski's T=10 / loss_scale=100 (its ``methods._train_token_
    faithful`` defaults); every other compatible arm is plain forward-KL at
    T=1, loss_scale=1. lr is the shared 5e-3 default of all these trainers."""
    if not lockstep_compatible(arm):
        raise ValueError(
            f"{arm!r} is not lockstep-compatible: "
            f"{lockstep_incompatible_reason(arm)}")
    if arm == "vector_rkl":
        return LockstepArmParams(loss="rkl")
    if arm == "vector_js":
        return LockstepArmParams(loss="js")
    if arm == "token_faithful":
        return LockstepArmParams(temperature=10.0, loss_scale=100.0)
    if arm in ("psr_kl", "psr_kl_all"):
        # train_psr's reg_coeff default (0.1) rides through train_object_kl
        # on the solo path, so the lockstep step must carry the same term
        # or the bitwise contract breaks. tests/test_psr.py pins this
        # mirror against inspect.signature(train_psr).
        return LockstepArmParams(reg_coeff=0.1)
    return LockstepArmParams()


@dataclass
class LockstepArm:
    """One arm in a lockstep group: a ready-to-train object plus the params
    the shared step needs. The object exposes the harness object protocol
    (``trainable``, ``inject``, ``apply(model, scale)``)."""

    name: str
    obj: object
    params: LockstepArmParams = field(default_factory=LockstepArmParams)


#: The compatible arms whose object can honor a position-masked
#: (decode-only) fit: their apply() yields a live SteeringHandle — the
#: steering handle itself for the hooks.steering-backed vector arms
#: (vector_all included: MultiLayerVectorObject routes its fit leg into
#: hooks.steering, 2026-08-20), the vectorless PositionMaskedFit tracker
#: for the lora/psr hook objects (lora_r1_all included: MultiLayerLoRAObject
#: mirrors RankOneLoRAObject's tracker + torch.where gating, 2026-08-20) —
#: which train_lockstep uses to set per-row training boundaries AND to
#: derive the loss gather from the same mask. token_bos/token_faithful
#: stay out FOREVER (a soft PROMPT token has no generated-position notion).
_MASKABLE_ARMS = ("vector", "vector_rkl", "vector_js", "vector_all",
                  "lora_r1", "lora_r1_all", "lora_r4", "psr_kl",
                  "psr_kl_all")


def build_lockstep_object(arm: str, model, tok, instruction, pairs, *, layer: int,
                          sentinel=None, positions: str = "all",
                          response_header_len: int | None = None,
                          lora_rank: int | None = None):
    """Construct the ready-to-train object for a compatible ``arm``, mirroring
    the SETUP its registry trainer performs (identical constructors and init
    helpers), so a lockstep-trained object's INIT is bit-identical to the
    registry path's. Raises for an incompatible arm.

    Only the compatible arms are handled; each construction is the exact line
    ``methods.py`` runs before it calls ``train_object_kl`` / ``train_steered_
    lm`` (verified by the bitwise test against ``ARMS[arm].train``). ``sentinel``
    is forwarded to the soft-token arms (the same optional the registry trainers
    accept — production auto-finds one, tiny tokenizers pass an explicit one).

    ``positions``/``response_header_len`` select the fit's mask leg:
    ``"all"`` (default — the historical all-token fit) or ``"response"``
    (decode-only; header-steered when ``response_header_len`` is set). The
    ``_MASKABLE_ARMS`` honor it: the vector arms (vector_all included)
    through hooks.steering, lora_r1/lora_r1_all/psr_kl/psr_kl_all through
    their objects' PositionMaskedFit tracker (same SteeringHandle mask, same
    loss_gather_masks tie). The soft-token arms REFUSE it forever — a
    PROMPT-embedded sentinel has no generated-position notion; a silent
    all-token fit labelled decode-only is exactly the artifact the
    mask→loss wiring exists to prevent.

    ``lora_rank`` (re-tune@32: rank joined the lora arm's search grid
    {1, 4, 16}) parametrizes lora_r1_all's MultiLayerLoRAObject; ``None``
    keeps the constructor's own default (rank 1, bit-identical to the
    historical arm — multilayer.py's legacy-shape contract). Any OTHER arm
    given a rank refuses loudly: rank is the lora arm's axis, and a rank
    silently ignored on a psr arm is a configured intent that never
    reached the trainer (the 3dd9e36 bug class)."""
    if not lockstep_compatible(arm):
        raise ValueError(
            f"{arm!r} is not lockstep-compatible: "
            f"{lockstep_incompatible_reason(arm)}")
    if lora_rank is not None and arm not in ("lora_r1_all", "lora_r4"):
        raise ValueError(
            f"lora_rank={lora_rank} for arm {arm!r} — rank parametrizes "
            "the multi-layer lora arm's object only; refusing to silently "
            "ignore it")
    if positions not in ("all", "response"):
        raise ValueError(
            f"unsupported fit positions {positions!r} (known: 'all', "
            "'response')")
    if positions != "all" and arm not in _MASKABLE_ARMS:
        reasons = {
            "token_bos": (
                "its intervention is a soft PROMPT token (sentinel "
                "embedding right after BOS) — every position it can act "
                "on is a prompt position, so 'steer only the generated "
                "tokens' has no well-defined action for it"),
            "token_faithful": (
                "its intervention is a soft PROMPT token (sentinel "
                "embedding appended as the final prompt position) — every "
                "position it can act on is a prompt position, so 'steer "
                "only the generated tokens' has no well-defined action "
                "for it"),
        }
        reason = reasons.get(arm, (
            "its object's position mask is not wired; extend the object "
            "with hooks.PositionMaskedFit before fitting this leg"))
        raise ValueError(
            f"{arm!r} cannot honor a positions={positions!r} (decode-only) "
            f"fit: {reason} — never fit it all-token under a decode-only "
            "label.")
    # Local imports mirror methods.py's own imports without importing methods
    # (whose ARMS dict this module must not perturb).
    from .objects import RankOneLoRAObject, SoftTokenObject
    from .multilayer import MultiLayerLoRAObject, MultiLayerVectorObject
    from .psr import PSRObject, init_psr_directions
    from ..vectors import SteeringVector

    if arm == "vector_all":
        obj = MultiLayerVectorObject(model)
        if positions != "all":
            obj.set_fit_positions(positions, response_header_len)
        return obj
    if arm == "lora_r1":
        obj = RankOneLoRAObject(model, layer=layer)
        if positions != "all":
            obj.set_fit_positions(positions, response_header_len)
        return obj
    if arm in ("lora_r1_all", "lora_r4"):
        obj = (MultiLayerLoRAObject(model) if lora_rank is None
               else MultiLayerLoRAObject(model, rank=int(lora_rank)))
        if positions != "all":
            obj.set_fit_positions(positions, response_header_len)
        return obj
    if arm == "token_bos":
        return SoftTokenObject(model, tok, sentinel=sentinel)
    if arm == "token_faithful":
        from .tokens import AppendedSoftTokenObject
        ids = tok(instruction.text, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(
                f"instruction {instruction.id!r} tokenized to no tokens")
        init = model.get_input_embeddings().weight.detach().float()[ids].mean(0)
        return AppendedSoftTokenObject(model, tok, sentinel=sentinel, init=init)
    if arm in ("psr_kl", "psr_kl_all"):
        layers = (list(range(num_layers(model))) if arm == "psr_kl_all"
                  else [layer])
        obj = PSRObject(model, layers, train_direction=True)
        if arm == "psr_kl":
            # Single layer only: the diffmean vector is the standard
            # informed start. psr_kl_all keeps its ZERO init (exactly
            # mirroring methods._make_train_psr, which the bitwise test
            # holds this to): the all-layer diffmean init blew step-0 KL to
            # 9.28 vs a 0.148 no-op floor and gate-died on the first AdamW
            # step — see extraction/psr.py's module docstring.
            obj.set_directions(init_psr_directions(model, tok, pairs, layers))
        if positions != "all":
            obj.set_fit_positions(positions, response_header_len)
        return obj
    if arm in ("vector", "vector_rkl", "vector_js"):
        # The additive-vector arms train through train_steered_lm on a bare
        # nn.Parameter; wrap it in a minimal object exposing the harness
        # protocol so the one lockstep engine drives every arm uniformly. A
        # trained one saves as the SAME SteeringVector artifact the registry's
        # VectorObject writes, so it re-hydrates through methods.load_object.
        from ..models import hidden_size
        from torch import nn

        loss = {"vector": "soft", "vector_rkl": "rkl",
                "vector_js": "js"}[arm]

        class _VectorArmObject:
            kind = "vector"

            def __init__(self):
                self._vec = nn.Parameter(torch.zeros(hidden_size(model)))
                self._layer = layer
                # The fit's mask leg: apply() yields the SteeringHandle, so
                # train_lockstep sets per-row training boundaries on it and
                # derives the loss gather from the SAME mask (the
                # header-steered convention when response_header_len is set).
                self._positions = positions
                self._response_header_len = response_header_len

            @property
            def trainable(self):
                return [self._vec]

            def inject(self, rendered_prompt: str) -> str:
                return rendered_prompt

            def apply(self, m, scale: float = 1.0):
                from ..hooks import steering
                return steering(m, {self._layer: self._vec}, scale=scale,
                                positions=self._positions, skip_bos=True,
                                response_header_len=self._response_header_len)

            def to_steering_vector(self) -> SteeringVector:
                # The fit's mask leg travels in the sidecar metadata — the
                # SteeringVector convention (train_distill and train_bidpo
                # both record it): FIT provenance in the artifact, while
                # hydration stays a plain all-position VectorObject and the
                # DEPLOY leg remains the eval cell's explicit choice
                # (jailbreak._set_response_positions). Without this stamp a
                # lockstep-trained response-fit vector was byte-
                # indistinguishable from an all-token fit.
                return SteeringVector(
                    vectors={self._layer: self._vec.detach().float().cpu()},
                    method=f"distill_{loss}", model_name="", instruction_id="",
                    metadata={"positions": self._positions,
                              "response_header_len":
                                  self._response_header_len})

            def to_(self, device) -> None:
                self._vec.data = self._vec.data.to(device)

            def save(self, path) -> None:
                self.to_steering_vector().save(path)

        return _VectorArmObject()
    raise ValueError(f"no lockstep builder for compatible arm {arm!r}")


def _batches_equal(a: Batch, b: Batch) -> bool:
    return (a.input_ids.shape == b.input_ids.shape
            and torch.equal(a.input_ids, b.input_ids)
            and torch.equal(a.attention_mask, b.attention_mask)
            and torch.equal(a.labels, b.labels))


def _encode_arm_batches(tok, obj, examples, *, max_len: int, batch_size: int,
                        label: str | None = None):
    """Build one arm's (teacher, student) batch list exactly as
    ``train_object_kl`` does: ``obj.inject`` on the student prompt, teacher
    side un-injected, ``_encode_aligned`` per example, then chunk in ORDER.

    ``label`` names the phase for the tokenize progress line; None stays
    silent (unit tests and tiny encodes should not narrate)."""
    prog = (TokenizeProgress(len(examples), label=label, max_len=max_len)
            if label else None)
    encoded = []
    for ex in examples:
        inj = DistillExample(
            prompt_with_si=ex.prompt_with_si,
            prompt_without=obj.inject(ex.prompt_without),
            completion=ex.completion,
            instruction_idx=ex.instruction_idx,
        )
        encoded.append(_encode_aligned(tok, inj, max_len))
        if prog is not None:
            prog.tick(len(encoded))
    if prog is not None:
        prog.done()
    teacher_batches, student_batches = [], []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        teacher_batches.append(collate([t for t, _ in chunk], tok.pad_token_id))
        student_batches.append(collate([s for _, s in chunk], tok.pad_token_id))
    return teacher_batches, student_batches


def _steering_handle(ctx_value):
    """The live SteeringHandle an arm's ``apply()`` yielded, or ``None``
    when the ctx yields nothing/None — an object fitting all-token (the
    soft tokens always; the PositionMaskedFit objects on their default
    leg), i.e. the historical every-position convention."""
    from ..hooks import SteeringHandle

    return ctx_value if isinstance(ctx_value, SteeringHandle) else None


def _masked_student_pass(model, arm: LockstepArm, student_b: Batch):
    """One student forward under the arm's ctx, with the loss gather derived
    from THE SAME steering handle the hook consumed (``loss_gather_masks`` —
    the mask→loss wiring). For a boundary-mode handle the per-row training
    boundary is set from the student batch's prompt lengths BEFORE the
    forward (never derived from the padded batch width — the trap). Grad
    mode is the caller's. Returns ``(logits, s_gather, keep, reg)`` — reg
    is the arm's anti-dead-gate regularizer TERM (coefficient applied by
    the caller) when ``arm.params.reg_coeff`` is set, else None; it must be
    computed here, inside the ctx, because the PSR gate stash it reads is
    cleared at context exit, and via the object's ONE shared assembly
    (``gate_reg_term``) so the solo path (train_object_kl) computes the
    bitwise-identical term."""
    with arm.obj.apply(model, scale=1.0) as ctx_value:
        handle = _steering_handle(ctx_value)
        if handle is not None and handle.positions in ("response", "prompt"):
            handle.set_prompt_lens(
                steer_prompt_boundaries(student_b,
                                        handle.response_header_len))
        logits = model(input_ids=student_b.input_ids,
                       attention_mask=student_b.attention_mask).logits
        s_gather, _t_gather, keep = loss_gather_masks(handle, student_b)
        reg = None
        if arm.params.reg_coeff:
            if not hasattr(arm.obj, "gate_reg_term"):
                raise ValueError(
                    f"{arm.name!r}: reg_coeff={arm.params.reg_coeff} but "
                    f"{type(arm.obj).__name__} exposes no gate_reg_term — "
                    "the anti-dead-gate regularizer is a PSR-gate "
                    "mechanism; do not set reg_coeff for an arm without a "
                    "gate.")
            reg = arm.obj.gate_reg_term(student_b.attention_mask, handle)
    return logits, s_gather, keep, reg


def _masked_student_step(model, arm: LockstepArm, student_b: Batch,
                         shared_t_logits, opt, *, zero_grad: bool = True,
                         w_div: float = 1.0, w_reg: float = 1.0):
    """One TRAINING micro-step's forward + loss + BACKWARD, all inside the
    arm's ``apply()`` context. The backward must run while the steering
    hooks are still registered: under activation checkpointing (the Qwen
    16384 fits) backward RECOMPUTES each layer's forward, and a recompute
    without the hooks is a different graph — torch's non-reentrant
    checkpoint refuses it (CheckpointError; diagnosed op-by-op
    2026-08-27: the recompute stream was missing exactly the hook's
    ops, because the historical callers exited apply() before calling
    backward). Moving backward inside the context is math-neutral
    without checkpointing — hooks only shape the FORWARD graph, and
    traversing an already-built graph is placement-independent — and it
    is what makes checkpointed gradients equal uncheckpointed ones
    (tests/test_ckpt_grad_parity.py pins that per arm).

    ``shared_t_logits`` is the online gathered ``[N, V]`` tensor or a
    precomputed top-K ``(idx, val)`` head (see :func:`_arm_divergence`).

    Gradient accumulation (``accum_steps`` > 1) drives the three keyword
    knobs: ``zero_grad`` only on a group's FIRST micro-batch, ``w_div`` /
    ``w_reg`` the micro-batch's completion-token / row share of its group
    (each part weighted by its OWN loss denominator, so the accumulated
    gradient equals one true batch's — see LockstepArmParams). The
    optimizer step belongs to the CALLER at group end (clip once on the
    accumulated gradient, then step). At the defaults (zero_grad=True,
    weights 1.0, weight multiplies skipped) the op sequence is
    byte-identical to the historical per-batch step.

    Op order is identical to the historical caller: forward, gathers,
    reg, teacher subset, divergence, +reg, zero_grad, backward — only
    the context-exit point moved. Returns ``(step_loss, reg)`` detached
    (the WEIGHTED contribution under accumulation), for the caller's
    history/logging."""
    with arm.obj.apply(model, scale=1.0) as ctx_value:
        handle = _steering_handle(ctx_value)
        if handle is not None and handle.positions in ("response", "prompt"):
            handle.set_prompt_lens(
                steer_prompt_boundaries(student_b,
                                        handle.response_header_len))
        logits = model(input_ids=student_b.input_ids,
                       attention_mask=student_b.attention_mask).logits
        s_gather, _t_gather, keep = loss_gather_masks(handle, student_b)
        reg = None
        if arm.params.reg_coeff:
            if not hasattr(arm.obj, "gate_reg_term"):
                raise ValueError(
                    f"{arm.name!r}: reg_coeff={arm.params.reg_coeff} but "
                    f"{type(arm.obj).__name__} exposes no gate_reg_term — "
                    "the anti-dead-gate regularizer is a PSR-gate "
                    "mechanism; do not set reg_coeff for an arm without a "
                    "gate.")
            reg = arm.obj.gate_reg_term(student_b.attention_mask, handle)
        s_logits = _gather_completion_logits(logits, student_b,
                                             gather=s_gather)
        if keep is None or bool(keep.all()):
            t_logits = shared_t_logits
        elif isinstance(shared_t_logits, tuple):
            t_logits = (shared_t_logits[0][keep], shared_t_logits[1][keep])
        else:
            t_logits = shared_t_logits[keep]
        if (w_div != 1.0 and keep is not None and not bool(keep.all())):
            # The token-share weight was derived from the label mask; a
            # keep that drops supervised positions means batchmean's true
            # denominator is smaller and the group weights would be wrong.
            # Unreachable for the production legs (all-token, or
            # header-steered response where a dropped position already
            # raises in loss_gather_masks) — refuse rather than mis-weight.
            raise ValueError(
                f"{arm.name!r}: accum_steps > 1 with a partially-masked "
                "fit (the steering mask dropped supervised completion "
                "positions) is unsupported — the accumulation weights "
                "assume the loss reads every completion position.")
        step_loss = _arm_divergence(arm.params, t_logits, s_logits)
        if w_div != 1.0:
            step_loss = step_loss * w_div
        if reg is not None:
            step_loss = step_loss + arm.params.reg_coeff * (
                reg if w_reg == 1.0 else w_reg * reg)
        if zero_grad:
            opt.zero_grad()
        step_loss.backward()
    return (step_loss.detach(), reg.detach() if reg is not None else None)


def _arm_divergence(params: LockstepArmParams, t_logits,
                    s_logits: torch.Tensor) -> torch.Tensor:
    """The arm's own scaled soft divergence on gathered logit pairs.

    ``t_logits`` is either the online teacher's ``[N, V]`` gathered logits
    or a precomputed top-K head ``(idx [N, K] int32, val [N, K])`` tuple
    (:mod:`.teacher_topk`) — the latter routes through
    :func:`distill.soft_divergence_topk` (forward KL only; train_lockstep
    refuses top-K consumption for any other loss/temperature at setup, so
    reaching here misconfigured is a programming error and raises)."""
    if isinstance(t_logits, tuple):
        from .teacher_topk import AUDITED_TOPK_TEMPERATURES

        if (params.loss != "soft"
                or params.temperature not in AUDITED_TOPK_TEMPERATURES):
            raise ValueError(
                f"precomputed top-K teacher heads carry only the forward-KL "
                f"objective at the audited temperature grid "
                f"{AUDITED_TOPK_TEMPERATURES} (loss={params.loss!r}, "
                f"T={params.temperature}) — train_lockstep's setup gate "
                "should have refused this arm")
        t_idx, t_val = t_logits
        return params.loss_scale * soft_divergence_topk(
            t_idx, t_val, s_logits, temperature=params.temperature)
    t_logp = torch.log_softmax(t_logits.float() / params.temperature, -1)
    s_logp = torch.log_softmax(s_logits.float() / params.temperature, -1)
    return params.loss_scale * soft_divergence(params.loss, t_logp, s_logp)


def _arm_divergence_per_example(params: LockstepArmParams,
                                t_logits: torch.Tensor,
                                s_logits: torch.Tensor,
                                counts) -> list[float]:
    """The arm's own scaled soft divergence, PER EXAMPLE: per-token values
    (:func:`distill.soft_divergence_per_token`, same temperature/scale as
    :func:`_arm_divergence`) grouped back into per-example token-means by
    the gather's per-row counts. The chunking-invariant unit the probe
    estimators average over — a mean of per-batch batchmeans moved when
    ``batch_size`` (a memory knob) re-chunked the same examples."""
    t_logp = torch.log_softmax(t_logits.float() / params.temperature, -1)
    s_logp = torch.log_softmax(s_logits.float() / params.temperature, -1)
    per_tok = soft_divergence_per_token(params.loss, t_logp, s_logp)
    return [params.loss_scale * v
            for v in per_example_means(per_tok, counts)]


def _probe_pair(model, arm: LockstepArm, t_full: torch.Tensor,
                student_b: Batch, *, with_noop: bool = True
                ) -> tuple[list[float], list[float] | None]:
    """Per-example (active, no-op) losses for ``arm`` on one probe batch,
    both legs on the ACTIVE leg's position support (so a masked fit's
    floor is measured on the same positions its loss trains). ``t_full``
    is the batch's plain completion-gathered teacher logits. No grad, no
    RNG draws, no param writes — the lockstep bitwise contract is
    untouched. PURE divergence on both legs, regularizer excluded: the
    no-op leg has no gates to pool (object inactive), so including the
    term on the active leg only would skew the very ratio the floor guard
    rules on — and the term is identically zero at the wide-open init
    anyway.

    Returns PER-EXAMPLE value lists (batch row order), not batch scalars:
    the estimators upstream average examples, so the estimate cannot move
    when the same examples are re-chunked at a different ``batch_size``.

    ``with_noop=False`` skips the object-inactive student forward and
    returns ``(active, None)`` — for the end-of-training probe, whose
    floor was already measured at startup (the base is frozen, so the
    no-op leg cannot have moved)."""
    with torch.no_grad():
        act_logits, s_gather, keep, _reg = _masked_student_pass(model, arm,
                                                                student_b)
        t_logits = t_full if keep is None or bool(keep.all()) else t_full[keep]
        counts = s_gather.sum(1)

        def _leg(logits):
            s_logits = _gather_completion_logits(logits, student_b,
                                                 gather=s_gather)
            return _arm_divergence_per_example(arm.params, t_logits,
                                               s_logits, counts)

        act = _leg(act_logits)
        noop = None
        if with_noop:
            inact_logits = model(
                input_ids=student_b.input_ids,
                attention_mask=student_b.attention_mask).logits
            noop = _leg(inact_logits)
    return act, noop


def arm_probe_losses(model, tok, obj, examples, *, max_len: int,
                     batch_size: int = 2,
                     n_probe: int | None = DEFAULT_PROBE_N,
                     params: LockstepArmParams | None = None,
                     families: list[str] | None = None,
                     ) -> ProbeEstimate:
    """Mean (active, no-op) soft-divergence probe over up to ``n_probe``
    examples, for ANY object exposing the harness protocol (``inject`` +
    ``apply``). Returns a :class:`ProbeEstimate` — a 2-tuple ``(active,
    noop)`` carrying sd/sem/counts/selection as attributes.

    The uniform end-of-training floor probe for arms trained OUTSIDE the
    shared KL loop: their own objectives differ (activation MSE, CE on
    teacher tokens, closed-form fits), but "did the object move the
    unprompted student TOWARD the SI-prompted teacher, or away" is
    measurable for every object, and the KL-vs-no-op-floor ratio is this
    programme's canonical degeneration signal (the 966-step silent failure
    WAS a KL plateau: 4.82 against a 0.148 floor). Pair with
    ``distill.floor_ratio_check``.

    DEFAULT CHANGE (2026-08-24, deliberate and loud): this probe used to
    read BATCH 0 ONLY — ``batch_size`` examples, in practice the arm's
    training batch size, i.e. every recorded floor and ratio of the 37-GPU
    wave was a 2-4 sample estimate of a KL. Measured against the true
    floors (see ``DEFAULT_PROBE_N``'s comment in distill.py), batch-0
    floors were off by up to 2x (si_irrelevant), two independent small-n
    estimators ordered the three SIs in OPPOSITE directions, and an arm
    was rejected at 3.14x whose absolute fit was BETTER than one accepted
    at 2.63x. The knobs are DECOUPLED:

    - ``n_probe`` — how many examples the ESTIMATE uses (statistics).
      Clamped to the pool (a 3-example fixture probes 3, never crashes,
      never over-samples); ``None`` = every example given.
    - ``batch_size`` — how many examples share one no-grad forward
      (memory). Re-chunking the same examples leaves the estimate
      untouched (per-example units — see :func:`_probe_pair`); it neither
      sets the sample size nor tilts the mean any more.

    SELECTION (2026-08-24, the second half of the same failure): when
    ``n_probe`` is smaller than the pool, the subset is chosen by
    sha256-rank of each example's student rendering (``prompt_without`` —
    SI-invariant, so per-SI probes over matched pools stay PAIRED), never
    ``pool[:n]``: the driver's pools arrive sorted by (family, goal_id,
    variant_key), so a prefix is a prefix of the alphabetically first
    family — the measured "256-prompt" floors were in fact 235 aim + 21
    amplegcg_plus. The selected keys' sha8 prefixes ride on the estimate
    (``selected_sha8``) so every recorded floor names its population.

    ``families`` (optional, parallel to ``examples``): per-family labels.
    When given, selection STRATIFIES — ``ceil(n_probe / n_families)`` per
    family by within-family sha256 rank, so thin families are estimable
    rather than absent — and the estimate carries a ``per_family`` table
    (active, noop, ratio, n per family): the numbers commensurable with
    per-family attack-success rates. The realised total can exceed
    ``n_probe`` by up to one row per family (recorded in ``n_examples``).

    AUTHORITATIVE AND UNCACHED — deliberately no ``teacher_prefix_cache``
    knob: every consumer of this probe is DECISION-BEARING — the driver's
    floor_ratio_max guard (a rejection writes NO artifact, so a
    cache-flipped verdict is unrecoverable) and lofit/loreft's per-epoch
    early stopping (which epoch's weights are kept). The teacher prefix
    cache (:mod:`.prefix_cache`) serves gradient-training forwards only;
    its bf16 kernel-shape drift must never reach a number a verdict rules
    on. Pinned by tests/test_prefix_cache.py (the probe never touches the
    cache).
    """
    params = params or LockstepArmParams()
    device = next(model.parameters()).device
    pool = list(examples)
    if not pool:
        raise ValueError("arm_probe_losses needs at least one example")
    if families is not None and len(families) != len(pool):
        raise ValueError(
            f"families has {len(families)} labels for {len(pool)} examples "
            "— they must be parallel")
    if n_probe is not None and int(n_probe) < 1:
        raise ValueError(
            f"n_probe must be >= 1, or None for every example given "
            f"(got {n_probe})")
    keys = [ex.prompt_without for ex in pool]
    if n_probe is None or int(n_probe) >= len(pool):
        idx = list(range(len(pool)))
    elif families is None:
        idx = sha_rank_indices(keys, int(n_probe))
    else:
        n_fams = len(set(families))
        idx = stratified_sha_rank_indices(
            keys, families, -(-int(n_probe) // n_fams))
    pool = [pool[i] for i in idx]
    sel_fams = None if families is None else [families[i] for i in idx]
    selected_sha8 = tuple(sha_hex(keys[i])[:8] for i in idx)
    bs = max(min(int(batch_size), len(pool)), 1)
    t_batches, s_batches = _encode_arm_batches(
        tok, obj, pool, max_len=max_len, batch_size=bs)
    arm = LockstepArm("probe", obj, params)
    acts: list[float] = []
    noops: list[float] = []
    for tb, sb in zip(t_batches, s_batches):
        teacher_b, student_b = tb.to(device), sb.to(device)
        with torch.no_grad():
            t_full = _gather_completion_logits(
                model(input_ids=teacher_b.input_ids,
                      attention_mask=teacher_b.attention_mask).logits,
                teacher_b,
            )
        act, noop = _probe_pair(model, arm, t_full, student_b)
        acts.extend(act)
        noops.extend(noop)
    act_m, act_sd, act_sem = _mean_sd_sem(acts)
    noop_m, noop_sd, noop_sem = _mean_sd_sem(noops)
    return ProbeEstimate(act_m, noop_m,
                         active_sd=act_sd, active_sem=act_sem,
                         noop_sd=noop_sd, noop_sem=noop_sem,
                         ratio_sem=_ratio_sem(acts, noops),
                         n_examples=len(pool), n_batches=len(t_batches),
                         batch_size=bs,
                         active_values=tuple(acts), noop_values=tuple(noops),
                         selected_sha8=selected_sha8,
                         per_family=(None if sel_fams is None else
                                     probe_family_table(sel_fams, acts,
                                                        noops)))


def resolve_warmup_steps(warmup_frac: float, total_steps: int) -> int:
    """The ONE warmup_frac -> steps policy, shared by every trainer and
    driver that resolves a fractional warmup against its own step budget
    (``max(1, round(frac * total))`` — the rounding and the >=1 floor are
    load-bearing at small step counts)."""
    return max(1, int(round(float(warmup_frac) * int(total_steps))))


def _lr_at_factory(lr: float, schedule: str, warmup_steps: int,
                   total_steps: int) -> Callable[[int], float]:
    """train_psr's mse-loop schedule, formula for formula (same warmup ramp
    ``lr*(i+1)/warmup_steps``, same cosine decay to 0 over the FULL step
    budget — early stopping may end an arm sooner; the horizon is the cap),
    so 'warmup_cosine' means one thing across trainers. 0-based step
    index."""
    def _lr_at(i: int) -> float:
        if schedule == "constant":
            return lr
        if i < warmup_steps:
            return lr * (i + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return lr
        frac = (i - warmup_steps) / (total_steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * frac))

    return _lr_at


def train_lockstep(
    model,
    tok,
    arms: list[LockstepArm],
    examples: list[DistillExample],
    *,
    epochs: int = 6,
    batch_size: int = 2,
    max_len: int = 768,
    seed: int = 0,
    log_fn=None,
    heldout: list[DistillExample] | None = None,
    early_stop: EarlyStopSpec | None = None,
    diagnostics: dict | None = None,
    n_probe: int | None = DEFAULT_PROBE_N,
    teacher_prefix_cache: bool = False,
    probe_cb: Callable[[int, str, object, float], None] | None = None,
    teacher_topk=None,
    checkpoint_every: int | None = None,
    checkpoint_dir=None,
    checkpoint_key: str | None = None,
    divergence_guard=None,
    epoch_cb: Callable[[int, int, list], None] | None = None,
) -> dict[str, list[float]]:
    """Train every arm in ``arms`` in lockstep, sharing one teacher forward per
    batch. Objects are trained in place; returns ``{arm.name: loss_history}``.

    Bitwise-equivalent to training each arm solo through its registry trainer
    at the same ``epochs``/``batch_size``/``max_len``/``seed`` (and matching
    per-arm ``LockstepArmParams``). ``epochs``/``batch_size``/``max_len``/
    ``seed`` are SHARED across the group — they must be, so every arm's teacher
    batches align and the single shared forward is valid for all of them; the
    per-arm ``lr``/``loss``/``temperature``/``loss_scale`` are read from each
    arm's ``params``. ONE documented exception: an arm whose params opt into
    the optimiser schedule (``lr_schedule``/``warmup_steps``/
    ``max_grad_norm`` — see :class:`LockstepArmParams`) deliberately departs
    from its solo trainer's fixed-lr loop; the solo==lockstep equivalence
    holds at the default (no-schedule) params, and a scheduled arm's own
    optimiser and RNG stream keep its schedule from perturbing any partner.

    Mask→loss wiring: each arm's student forward runs under its own ctx; an
    arm whose ``apply()`` yields a SteeringHandle (the vector arms) gets its
    per-row training boundary set from the student batch's prompt lengths,
    and its loss gathers exactly the completion positions the handle's mask
    covers, with the SHARED teacher logits sub-selected to the same
    completion indices — so a masked (decode-only) arm and an all-token arm
    can share one teacher forward without the masked arm's loss reading
    unsteered positions.

    ``epoch_cb(epoch, step, per_arm)``: called after every epoch's held-out
    pass with the lockstep's per-arm state (``pa["arm"].obj``, ``history``,
    ``heldout_curve``, ``stopped``, ``epochs_run``); the fit writers use it to
    save durable epoch-boundary artefacts. None = off.

    ``heldout`` + ``early_stop``: PER-ARM early stopping inside the shared
    loop. After every epoch each live arm's held-out loss (its own
    divergence/temperature/scale, object active, no grad) is evaluated on
    the shared held-out batches — ONE teacher forward per held-out batch per
    epoch feeds every arm — and an arm that stops improving
    (:class:`EarlyStopSpec`) stops taking steps and is restored to its BEST
    epoch, while the rest of the group trains on. An arm's stop cannot
    perturb another arm's trajectory (own optimiser, own RNG stream, shared
    batch order drawn regardless), so the bitwise contract holds per arm at
    its own stopping epoch. ``epochs`` is the cap.

    ``diagnostics`` (optional dict) is filled per arm with the startup/no-op
    floor probes, the end-of-training probe (same examples, trained object)
    and its floor ratio — each a MEAN over ``n_probe`` examples with its
    per-batch sd/sem and realised counts recorded alongside — the held-out
    curve, and the chosen stopping epoch. The guard itself
    (``distill.floor_ratio_check``) is NOT raised from here — a single
    arm's degeneration must never discard the group's other arms — the
    caller enforces it per arm before saving.

    ``n_probe`` (default ``DEFAULT_PROBE_N`` = 256; ``None`` = every
    training example) is the probes' SAMPLE SIZE, clamped to the pool.
    It used to be implicitly ``batch_size`` — batch 0 only, a 2-4 sample
    estimate of a KL whose within-SI spread exceeded the between-SI gaps
    the wave read as a finding (see ``arm_probe_losses``). Probe examples
    are selected by sha256-rank of the student rendering (never a pool
    prefix — the pools arrive family-sorted, so a prefix is one family)
    and their sha8 prefixes are recorded in the diagnostics; per-example
    accumulation makes the estimate invariant to ``batch_size``. The
    probes are no-grad, no-RNG, no-write, so widening them cannot perturb
    the bitwise contract.

    ``probe_cb`` (the production monitoring seam):
    ``probe_cb(step, arm_name, arm_obj, loss_val)``, invoked once
    per LIVE arm per optimizer step, immediately after that arm's
    ``opt.step()`` (its .grad tensors are still present and already
    clipped) and after its log record. The callback owns its own cadence
    (probes are expensive; ``fitmon.ProbeCadence`` gates them). If it
    raises ``fitmon.TrainingDiverged``, THAT ARM is aborted — marked
    stopped, its ``diagnostics`` entry carries ``diverged`` — while its
    group partners train on (per-arm optimiser/RNG isolation, same
    argument as early stopping); any other exception propagates. The
    default ``None`` leaves this loop byte-identical to the historical
    one: the floor-probe/holdout machinery is bitwise-untouched when
    monitors are off (pinned by tests/test_lockstep.py).

    ``teacher_prefix_cache``: every teacher batch of the group shares
    the SI's rendered prefix (~95% of the teacher tokens for the long
    SIs), so under the flag the prefix runs through the model ONCE per
    group and every GRADIENT-TRAINING shared teacher forward replays it
    as past_key_values (:mod:`.prefix_cache`). That is the cache's WHOLE
    reach: the startup/final probes and every held-out epoch-selection
    pass are DECISION-BEARING and stay AUTHORITATIVE AND UNCACHED, flag
    or no flag — see ``_authoritative_teacher_logits`` below and the
    cache-call-count tests in tests/test_prefix_cache.py. The prefix is
    derived from the FIT teacher batches, matching how a solo trainer
    derives its own cache from the same fit examples — so the solo ==
    lockstep bitwise contract HOLDS UNDER THE FLAG (both sides run
    identical cached kernels on identical batches; pinned flag-on by
    tests/test_prefix_cache.py, flag-off by test_lockstep's solo==
    lockstep suite). Cached-vs-uncached is fp-tolerance equivalence,
    not bitwise (kernel shapes differ — the row-sharding contract's
    accepted class); the two-SI config's ``teacher_prefix_cache: false``
    is the escape hatch, and this default ``False`` keeps every caller
    outside the two-SI driver byte-identical.

    ``teacher_topk`` (offline precompute, 2026-08-27): a
    :class:`~sisteer.extraction.teacher_topk.TeacherTopK` store. When
    given AND its rows bind to this fit's teacher batches, the
    GRADIENT-TRAINING teacher leg is not run at all — no shared teacher
    forward, no prefix cache — and each step's loss consumes the stored
    top-K head through ``distill.soft_divergence_topk`` (the exact,
    documented top-K forward KL). Setup REFUSES (raises) any arm whose
    loss is not plain T=1 forward KL — rkl/js need mass a head cannot
    carry, and at token_faithful's T=10 the head's mass collapses, so
    consuming it there would silently train a different objective than
    the HPO tuned (the caller routes such groups to the online teacher
    instead). Row binding failure (a store built from a different pool /
    tokenizer / truncation) FALLS BACK to the online teacher with a loud
    ``[teacher-topk] FALLBACK`` line — stale-artifact-shaped, so it
    degrades loudly rather than erroring a whole wave. DECISION-BEARING
    passes are UNAFFECTED either way: the startup/final floor probes and
    every held-out epoch-selection pass keep their authoritative,
    uncached, FULL-VOCAB online forwards, so no guard verdict or restored
    epoch can shift with the flag (the prefix-cache rule, inherited).

    Gradient accumulation rides per arm via
    ``LockstepArmParams.accum_steps`` (see its field note): groups of
    consecutive micro-batches along the shared epoch order, per-denominator
    share weights (completion tokens for the divergence, rows for the gate
    regularizer) so the accumulated gradient equals one true batch's, ONE
    clip on the accumulated gradient, one optimizer step, one history/log
    record and one ``probe_cb`` per GROUP; the lr schedule ticks and
    resolves warmup against OPTIMIZER steps (epochs x ceil(batches /
    accum_steps)). A ragged final group normalizes over its own members
    (bidpo's ceil-path contract). ``accum_steps=1`` is byte-identical to
    the historical loop.

    ``checkpoint_every``/``checkpoint_dir``/``checkpoint_key`` — the
    mid-fit durability seam (fit-wave hardening, 2026-08-27; see
    :mod:`.fitckpt` for the design points). Default OFF: with all three
    ``None`` this loop is byte-identical to before the seam existed (the
    fit driver opts in; HPO trials and every other caller are untouched).
    When on, a checkpoint is written every ~``checkpoint_every``
    optimizer steps — only at points where EVERY live arm has just
    completed an optimizer step (never inside an accumulation group), so
    resume carries no partial-group state — with the FULL trainer state:
    per-arm trainables, optimizer state, per-arm RNG stream, history,
    early-stop state (best/patience/snapshot), the startup-probe record,
    the epoch's shuffled order + position + order-RNG state, and torch
    CPU/CUDA RNG states. On start the seam loads the newest checkpoint
    whose ``checkpoint_key`` matches (the caller passes the cell's fit
    fingerprint); a mismatch or unreadable file is IGNORED loudly (fresh
    start), never a crash, never a partial restore. Setup (encoding,
    schedule resolution, startup probes, the epoch -1 held-out baseline)
    runs identically on a resume — those passes are deterministic
    no-grad recomputations of the same values — and the restore then
    overwrites every trajectory-bearing field, so a resumed fit's
    remaining steps are bitwise the uninterrupted fit's on CPU fp32
    (tests/test_fit_checkpoint.py pins interrupt/resume parity, an
    epoch-boundary crossing, active clipping, ragged batches, early-stop
    survival, and the key-mismatch fresh start). The checkpoint write
    serializes state only at checkpoint moments — no new per-step
    syncs — and moves no forward/backward code relative to the
    ``obj.apply()`` hook context.

    ``divergence_guard`` — the STEP-level divergence abort
    (:class:`fitmon.StepDivergenceSpec` documents the clauses): default
    ON (``None``/``True`` = production spec; ``False`` = off; a spec
    instance = custom tuning). Per-epoch early stopping rules once per
    epoch — ~8h on the slow fits — while a genuinely exploding arm is
    visible in its step losses within minutes; the guard feeds each arm's
    ALREADY-SYNCED per-step loss float (zero new syncs, no RNG, no
    writes — a healthy run's trajectory is bit-identical guard-on vs
    off) into per-arm state and, on a verdict, marks THAT ARM diverged
    exactly like a ``probe_cb`` ``TrainingDiverged`` (recorded failed,
    ``diagnostics`` carries the ``step-divergence: ...`` reason, no
    artifact ever ships) while its group partners train on.
    tests/test_divergence_guard.py pins: a diverging config dies in
    steps, a healthy config runs to completion bitwise-untouched.
    """
    if not arms:
        raise ValueError("train_lockstep needs at least one arm")
    for a in arms:
        if not lockstep_compatible(a.name):
            raise ValueError(
                f"{a.name!r} cannot be lockstep-trained: "
                f"{lockstep_incompatible_reason(a.name)} — train it solo")
        if a.params.loss not in SOFT_LOSSES:
            raise ValueError(
                f"{a.name!r} lockstep loss {a.params.loss!r} is not one of "
                f"the teacher-logit-distribution losses {SOFT_LOSSES}")
        if a.params.lr_schedule not in (None, "warmup_cosine", "constant"):
            raise ValueError(
                f"{a.name!r}: lr_schedule must be None (fixed lr, the "
                f"default), 'warmup_cosine' or 'constant' — got "
                f"{a.params.lr_schedule!r}")
        if a.params.warmup_steps is not None and a.params.lr_schedule is None:
            raise ValueError(
                f"{a.name!r}: warmup_steps={a.params.warmup_steps} without "
                "an lr_schedule — refusing to silently ignore it (set "
                "lr_schedule='warmup_cosine' or 'constant')")
        if a.params.warmup_frac is not None:
            # Mirrors the warmup_steps-without-schedule refusal above: a
            # warmup that reads as configured and behaves as nothing is
            # the 3dd9e36 class of bug.
            if a.params.lr_schedule is None:
                raise ValueError(
                    f"{a.name!r}: warmup_frac={a.params.warmup_frac} "
                    "without an lr_schedule — refusing to silently ignore "
                    "it (set lr_schedule='warmup_cosine' or 'constant')")
            if a.params.warmup_steps is not None:
                raise ValueError(
                    f"{a.name!r}: warmup_frac={a.params.warmup_frac} AND "
                    f"warmup_steps={a.params.warmup_steps} — they are "
                    "mutually exclusive (the fraction exists so the fit "
                    "re-derives steps from its own budget; a copied step "
                    "count next to it is a contradiction)")
            if not (0.0 < float(a.params.warmup_frac) < 1.0):
                raise ValueError(
                    f"{a.name!r}: warmup_frac must be in (0, 1), got "
                    f"{a.params.warmup_frac}")
        if int(a.params.accum_steps) < 1:
            raise ValueError(
                f"{a.name!r}: accum_steps must be >= 1, got "
                f"{a.params.accum_steps!r}")
        if teacher_topk is not None:
            from .teacher_topk import AUDITED_TOPK_TEMPERATURES

            if (a.params.loss != "soft"
                    or a.params.temperature not in AUDITED_TOPK_TEMPERATURES):
                raise ValueError(
                    f"{a.name!r}: a precomputed top-K teacher head carries "
                    f"only the forward-KL objective at the audited "
                    f"temperature grid {AUDITED_TOPK_TEMPERATURES} (the "
                    f"re-tune's swept grid, where the head KL IS the tuned "
                    f"objective), but this arm trains "
                    f"loss={a.params.loss!r} at T={a.params.temperature} — "
                    "consuming the head there would silently train an "
                    "objective nothing audited (token_faithful's T=10 head "
                    "mass collapses). Route this group to the online "
                    "teacher instead of passing teacher_topk.")

    checkpoint_every = resolve_checkpoint_kwargs(
        checkpoint_every, checkpoint_dir, checkpoint_key)
    from .fitmon import TrainingDiverged, resolve_divergence_guard

    device = next(model.parameters()).device
    # Frozen base — every solo trainer does this too; idempotent.
    for p in model.parameters():
        p.requires_grad_(False)

    # No completion truncation is what makes the ONE teacher forward valid for
    # every arm (differing student-prompt lengths — e.g. the +1 sentinel of a
    # token arm — must not shorten the shared completion). Refuse up front,
    # naming the number, if max_len cannot hold prompt + full completion.
    # Held-out examples ride the same shared teacher forwards, so they join
    # the same budget and preflight.
    all_examples = list(examples) + list(heldout or [])
    comp_budget = 1
    for ex in all_examples:
        n = len(cached_ids(tok, ex.completion, False))
        comp_budget = max(comp_budget, n + (tok.eos_token_id is not None))
    for a in arms:
        inj = [DistillExample(ex.prompt_with_si, a.obj.inject(ex.prompt_without),
                              ex.completion, ex.instruction_idx)
               for ex in all_examples]
        preflight_max_len(tok, inj, max_len, min_completion_tokens=comp_budget)

    # Per-arm batches, optimiser, RNG state. Each arm's own optimiser and RNG
    # stream => adding an arm cannot perturb another's trajectory.
    per_arm = []
    n_batches = None
    for a in arms:
        for p in a.obj.trainable:
            p.data = p.data.to(device)
        t_batches, s_batches = _encode_arm_batches(
            tok, a.obj, examples, max_len=max_len, batch_size=batch_size,
            label=f"{a.name} train")
        if n_batches is None:
            n_batches = len(t_batches)
        elif len(t_batches) != n_batches:
            raise ValueError(
                f"arm {a.name!r} produced {len(t_batches)} batches, group has "
                f"{n_batches} — every arm must share batch_size and examples")
        h_teacher, h_student = ([], [])
        if heldout:
            h_teacher, h_student = _encode_arm_batches(
                tok, a.obj, heldout, max_len=max_len, batch_size=batch_size,
                label=f"{a.name} held-out")
        opt = torch.optim.AdamW(a.obj.trainable, lr=a.params.lr,
                                weight_decay=a.params.weight_decay)
        # Own torch RNG stream, seeded like the arm's solo trainer. Saved/
        # restored around each step so no arm's draws touch another's — a
        # formality for these frozen-model steps (no RNG is drawn), kept so the
        # isolation is provable and future-proof.
        torch.manual_seed(seed)
        per_arm.append({
            "arm": a,
            "teacher": t_batches,
            "student": s_batches,
            "h_teacher": h_teacher,
            "h_student": h_student,
            "opt": opt,
            "rng_state": torch.get_rng_state(),
            "history": [],
            "heldout_curve": [],
            "stopper": (EarlyStopper(early_stop or EarlyStopSpec())
                        if heldout else None),
            "stopped": False,
            "epochs_run": 0,
            # Step-level divergence guard, per arm (default ON — see the
            # docstring). Observational: it reads the already-synced loss
            # float, so a healthy arm is bit-identical guard-on vs off.
            "divguard": resolve_divergence_guard(
                divergence_guard, tag=f"lockstep/{a.name}"),
        })

    # Per-arm optimiser schedule (opt-in via LockstepArmParams; lr_schedule
    # None = the historical fixed-lr loop and the step loop touches
    # NOTHING for that arm — bit-identity with today is the default, and
    # a scheduled arm's own optimiser/RNG stream means its schedule cannot
    # perturb a partner). Wired for psr_kl_all, the arm whose fixed-lr KL
    # fit ran away hardest in the wave (heldout 0.0151 -> 7.30 inside
    # epoch 1, 19-24/32 gates dead in epoch 0, final 34-38x the floor)
    # while its mse sibling got warmup/cosine/clipping in train_psr.
    for pa in per_arm:
        p = pa["arm"].params
        # Per-arm accumulation state: the arm's own OPTIMIZER-step budget
        # (groups of accum_steps micro-batches along the shared epoch
        # order; the ragged tail is its own smaller group), which is also
        # the horizon its lr schedule ticks and resolves warmup against.
        # At accum_steps=1 this is exactly the historical epochs*n_batches.
        pa["accum"] = int(p.accum_steps)
        total_steps = epochs * math.ceil(n_batches / pa["accum"])
        if pa["accum"] > 1:
            # The group-share weights' denominators, known before any
            # forward: completion-token count per micro-batch (batchmean's
            # denominator) and row count (gate_regularizer's denominator).
            pa["n_tok"] = [int((sb.labels[:, 1:] != -100).sum())
                           for sb in pa["student"]]
            pa["n_rows"] = [int(sb.input_ids.shape[0])
                            for sb in pa["student"]]
        if p.lr_schedule is None:
            pa["lr_at"] = None
            pa["resolved_warmup_steps"] = None
            continue
        warm = p.warmup_steps
        if p.warmup_frac is not None:
            # Resolved HERE, where this fit's own total step budget is
            # known — never a step count copied from a differently-sized
            # (subsampled) trial run.
            warm = resolve_warmup_steps(p.warmup_frac, total_steps)
        if warm is None:
            warm = min(500, max(1, total_steps // 10))  # train_psr's default
        pa["resolved_warmup_steps"] = int(warm)
        pa["lr_at"] = _lr_at_factory(p.lr, p.lr_schedule, int(warm),
                                     total_steps)

    # The shared held-out teacher forward is only valid if every arm's
    # held-out teacher batches are bit-identical — same assertion the
    # training loop makes per batch, made once up front here.
    if heldout:
        for pa in per_arm[1:]:
            for hbi, ht in enumerate(per_arm[0]["h_teacher"]):
                if not _batches_equal(pa["h_teacher"][hbi], ht):
                    raise ValueError(
                        f"held-out teacher batch {hbi} differs between "
                        f"{arms[0].name!r} and {pa['arm'].name!r} — the "
                        "shared held-out forward would be wrong for at "
                        "least one arm (completion truncation?)")

    # Precomputed top-K teacher heads (see the docstring): assembled ONCE
    # from arm 0's FIT teacher batches — the training loop asserts every
    # arm's teacher batch is bit-identical before each shared step, so arm
    # 0's rows stand for the group. Every row is bound by the sha of its
    # exact token ids (gather_for_batch); ANY binding failure means the
    # artifact describes a different world (pool, tokenizer, truncation),
    # so the whole group falls back to the online teacher, LOUDLY, and
    # trains exactly as if teacher_topk had never been passed.
    topk_batches = None
    if teacher_topk is not None:
        from .teacher_topk import TeacherTopKMismatch

        try:
            topk_batches = [teacher_topk.gather_for_batch(tb)
                            for tb in per_arm[0]["teacher"]]
        except TeacherTopKMismatch as e:
            topk_batches = None
            print(f"[teacher-topk] FALLBACK to the ONLINE teacher — the "
                  f"precomputed artifact does not bind to this fit's "
                  f"batches: {e}", flush=True)

    # Teacher prefix cache (see the docstring): built ONCE from arm 0's FIT
    # teacher batches — the training loop asserts every arm's teacher batch
    # is bit-identical before each shared forward, so arm 0's rows stand
    # for the group. GRADIENT-TRAINING FORWARDS ONLY (review finding
    # 2026-08-24): the startup/final probes and the held-out passes are
    # DECISION-BEARING — the probes produce the noop floor and the final
    # probe loss the floor_ratio_max bar rules on, and the held-out pass
    # selects each arm's restored epoch. A cache-induced bf16 shift there
    # can flip an accept/reject or pick a different epoch, and a rejected
    # arm writes NO artifact, so "re-run with the flag off" cannot recover
    # it after the fact. Those paths therefore stay AUTHORITATIVE AND
    # UNCACHED below, flag or no flag — pinned by the cache-call-count
    # tests in tests/test_prefix_cache.py. With a bound top-K store there
    # is no training-leg teacher forward to cache, so the prefix KV (and
    # its per-batch expansion memory) is skipped entirely.
    tpc = (prefix_cache_for_batches(model, per_arm[0]["teacher"])
           if teacher_prefix_cache and topk_batches is None else None)

    def _authoritative_teacher_logits(tb: Batch) -> torch.Tensor:
        """The plain, uncached full forward + completion gather — the ONE
        assembly for every DECISION-BEARING teacher pass (startup floor,
        held-out epoch selection, final probe). Deliberately ignores the
        cache: these numbers must be identical whether or not the flag is
        on, so a guard verdict or a restored epoch can never depend on
        cached-kernel fp drift."""
        with torch.no_grad():
            return _gather_completion_logits(
                model(input_ids=tb.input_ids,
                      attention_mask=tb.attention_mask).logits,
                tb,
            )

    def _train_teacher_logits(tb: Batch) -> torch.Tensor:
        """The batch's completion-gathered teacher logits for a TRAINING
        step — prefix-cached when the flag built a cache. Never used for a
        probe or held-out pass (the decision-bearing note above)."""
        if tpc is not None:
            return tpc.completion_logits(model, tb)
        return _authoritative_teacher_logits(tb)

    # Startup sanity check — one shared no-grad teacher pass per probe batch
    # plus two student passes per arm per batch (object active / inactive):
    # record each arm's step-0 loss next to its no-op floor and warn loudly
    # when an init starts far off the unsteered model (see
    # distill.startup_loss_check — how the diffmean-init all-layer PSR
    # silently gate-died at step 1). The no-op floor is kept per arm — MEAN,
    # per-EXAMPLE sd/sem, and the per-example floors themselves (for the
    # final ratio's covariance-aware sem): the end-of-training probe below
    # rules against it (distill.floor_ratio_check, enforced by the caller),
    # so it must be an ESTIMATE, not batch 0's 2-4 samples (the wave's
    # floors — see arm_probe_losses; si_irrelevant's read 2x the measured
    # truth). Probe examples are chosen by sha256-rank of the student
    # rendering, NEVER examples[:n]: the driver's pools arrive sorted by
    # (family, goal_id, variant_key), so the old take-the-first-batches
    # probe measured the alphabetically first family only — the same
    # single-family contamination arm_probe_losses documents. Probe batches
    # are encoded per arm (obj.inject on the student side, like training);
    # the teacher side is un-injected and hence identical across arms, so
    # arm 0's teacher batches stand for the group. No-grad, no RNG draws,
    # no param writes: the bitwise contract holds.
    if n_probe is not None and int(n_probe) < 1:
        raise ValueError(
            f"n_probe must be >= 1, or None for every training example "
            f"(got {n_probe})")
    probe_keys = [ex.prompt_without for ex in examples]
    probe_idx = sha_rank_indices(probe_keys,
                                 None if n_probe is None else int(n_probe))
    probe_examples = [examples[i] for i in probe_idx]
    probe_sha8 = [sha_hex(probe_keys[i])[:8] for i in probe_idx]
    probe_n_realised = len(probe_examples)
    probe_teacher_batches = None
    for pa in per_arm:
        pt, ps = _encode_arm_batches(tok, pa["arm"].obj, probe_examples,
                                     max_len=max_len, batch_size=batch_size,
                                     label=f"{pa['arm'].name} probe")
        if probe_teacher_batches is None:
            probe_teacher_batches = pt
        pa["probe_student"] = ps
    n_probe_batches = len(probe_teacher_batches)
    for bi in range(n_probe_batches):
        probe_teacher = probe_teacher_batches[bi].to(device)
        # AUTHORITATIVE AND UNCACHED: this pass measures the noop floor —
        # the denominator of every floor_ratio verdict.
        probe_t_logits = _authoritative_teacher_logits(probe_teacher)
        for pa in per_arm:
            s0, fl = _probe_pair(model, pa["arm"], probe_t_logits,
                                 pa["probe_student"][bi].to(device))
            pa.setdefault("probe_step0", []).extend(s0)
            pa.setdefault("probe_floor", []).extend(fl)
    for pa in per_arm:
        a = pa["arm"]
        step0, _s0_sd, _s0_sem = _mean_sd_sem(pa["probe_step0"])
        floor, floor_sd, floor_sem = _mean_sd_sem(pa["probe_floor"])
        pa["step0_loss"], pa["noop_floor"] = step0, floor
        pa["noop_floor_sd"], pa["noop_floor_sem"] = floor_sd, floor_sem
        startup_loss_check(step0, floor, tag=f"lockstep/{a.name}",
                           log_fn=log_fn)
        if log_fn:
            # The floor and its uncertainty, live (numeric scalars only —
            # the driver's metrics path filters to numbers regardless).
            rec = {f"lockstep/{a.name}/noop_floor": floor,
                   f"lockstep/{a.name}/probe_n": probe_n_realised}
            if floor_sem is not None:
                rec[f"lockstep/{a.name}/noop_floor_sem"] = floor_sem
            log_fn(rec, 0)

    def _heldout_pass(epoch: int, step: int) -> None:
        """Per-arm held-out loss for every LIVE arm — one shared teacher
        forward per held-out batch feeds them all — then each arm's
        stopper rules: a stopping arm is restored to its best epoch and
        takes no further steps. No grad, no RNG draws; a stop cannot
        perturb the others (own optimiser/RNG, batch order shared)."""
        live = [pa for pa in per_arm if not pa["stopped"]]
        if not live:
            return
        tot = {id(pa): [0.0, 0] for pa in live}
        # Per-batch values ride along for the sem ONLY; the stopper's metric
        # (hl below) keeps its exact token-weighted accumulation, bit for
        # bit, so early stopping is unchanged by the uncertainty reporting.
        per_b = {id(pa): [] for pa in live}
        n_h = len(per_arm[0]["h_teacher"])
        for hbi in range(n_h):
            h_teacher = per_arm[0]["h_teacher"][hbi].to(device)
            # AUTHORITATIVE AND UNCACHED: this pass selects each arm's
            # restored epoch (EarlyStopper.update below).
            ht_logits = _authoritative_teacher_logits(h_teacher)
            for pa in live:
                a = pa["arm"]
                h_student = pa["h_student"][hbi].to(device)
                with torch.no_grad():
                    # Held-out is a model-quality metric: pure divergence,
                    # regularizer excluded (a protective TRAINING term must
                    # not decide which epoch counts as best).
                    logits, s_gather, keep, _reg = _masked_student_pass(
                        model, a, h_student)
                    s_logits = _gather_completion_logits(
                        logits, h_student, gather=s_gather)
                    t = (ht_logits if keep is None or bool(keep.all())
                         else ht_logits[keep])
                    val = float(_arm_divergence(a.params, t, s_logits))
                k = int(s_gather.sum())
                tot[id(pa)][0] += val * k
                tot[id(pa)][1] += k
                per_b[id(pa)].append(val)
        for pa in live:
            a = pa["arm"]
            hl = tot[id(pa)][0] / max(tot[id(pa)][1], 1)
            pa["heldout_curve"].append(hl)
            if log_fn:
                rec = {f"lockstep/{a.name}/heldout_loss": hl,
                       "epoch": epoch}
                # The per-epoch probe's uncertainty, live next to its mean
                # (sem of the per-batch batchmeans; hl itself is the
                # token-weighted pool — identical for equal-size batches).
                _hm, _hsd, h_sem = _mean_sd_sem(per_b[id(pa)])
                if h_sem is not None:
                    rec[f"lockstep/{a.name}/heldout_sem"] = h_sem
                log_fn(rec, step)
            if epoch >= 0 and pa["stopper"].update(hl, a.obj.trainable):
                pa["stopper"].restore_best(a.obj.trainable)
                pa["stopped"] = True
                if log_fn:
                    log_fn({f"lockstep/{a.name}/stopped_epoch": epoch + 1,
                            f"lockstep/{a.name}/best_epoch":
                                pa["stopper"].best_epoch}, step)

    if heldout:
        # Pre-training baseline point (curve only — epoch -1 never updates
        # a stopper, so the init can never become the restore target; a fit
        # that only got worse must fail the floor guard loudly instead).
        _heldout_pass(-1, 0)

    # Shared batch order: one RNG seeded like every arm's solo ``random.Random
    # (seed)`` (the group requires a common seed), so the order matches each
    # arm's own shuffled order epoch for epoch.
    order_rng = random.Random(seed)
    step = 0

    # -- mid-fit checkpoint seam (default OFF — see the docstring) -----------
    def _save_ckpt(epoch_i: int, batches_done_i: int,
                   order_i: list[int]) -> None:
        """Serialize the FULL trainer state at an all-arms optimizer-step
        boundary (the only fire points, so no partial accumulation-group
        state exists to carry). GPU->CPU copies happen here and only here
        — the one-sync-per-cadence discipline between checkpoints is
        untouched."""
        arms_state = {}
        for pa in per_arm:
            a = pa["arm"]
            stopper_state = None
            if pa["stopper"] is not None:
                s = pa["stopper"]
                stopper_state = {
                    "best": s.best, "best_epoch": s.best_epoch,
                    "epochs_seen": s.epochs_seen,
                    "since_best": s._since_best,
                    "snapshot": (None if s._snapshot is None else
                                 [t.detach().cpu().clone()
                                  for t in s._snapshot]),
                }
            arms_state[a.name] = {
                "trainable": [p.detach().cpu().clone()
                              for p in a.obj.trainable],
                "opt_state": to_cpu(pa["opt"].state_dict()),
                "rng_state": pa["rng_state"],
                "history": list(pa["history"]),
                "heldout_curve": list(pa["heldout_curve"]),
                "stopped": pa["stopped"],
                "diverged": pa.get("diverged"),
                "epochs_run": pa["epochs_run"],
                "ep_pos": pa["ep_pos"],
                "stopper": stopper_state,
                # The startup-probe record: deterministic no-grad values a
                # resume recomputes identically, restored anyway so the
                # published diagnostics are the original run's bytes.
                "probe_step0": pa.get("probe_step0"),
                "probe_floor": pa.get("probe_floor"),
                "step0_loss": pa.get("step0_loss"),
                "noop_floor": pa.get("noop_floor"),
                "noop_floor_sd": pa.get("noop_floor_sd"),
                "noop_floor_sem": pa.get("noop_floor_sem"),
            }
        save_checkpoint(checkpoint_dir, {
            "checkpoint_key": checkpoint_key,
            "trainer": "train_lockstep",
            "arm_names": [a.name for a in arms],
            "n_batches": n_batches,
            "epoch": epoch_i,
            "batches_done": batches_done_i,
            "order": list(order_i),
            "order_rng_state": order_rng.getstate(),
            "step": step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": cuda_rng_states(),
            "arms": arms_state,
        })
        print(f"[fitckpt] train_lockstep checkpoint written: epoch "
              f"{epoch_i}, batch {batches_done_i}/{n_batches} in epoch, "
              f"optimizer steps "
              f"{max(len(pa['history']) for pa in per_arm)}", flush=True)

    start_epoch = 0
    resume_order: list[int] | None = None
    resume_batches_done = 0
    ckpt_last_steps = 0
    if checkpoint_every is not None:
        ck = load_checkpoint(
            checkpoint_dir, checkpoint_key,
            expect={"trainer": "train_lockstep",
                    "arm_names": [a.name for a in arms],
                    "n_batches": n_batches})
        if ck is not None:
            for pa in per_arm:
                s = ck["arms"][pa["arm"].name]
                with torch.no_grad():
                    for p, saved in zip(pa["arm"].obj.trainable,
                                        s["trainable"]):
                        p.data.copy_(saved.to(p.data.device))
                pa["opt"].load_state_dict(s["opt_state"])
                pa["rng_state"] = s["rng_state"]
                pa["history"] = list(s["history"])
                pa["heldout_curve"] = list(s["heldout_curve"])
                pa["stopped"] = bool(s["stopped"])
                if s.get("diverged"):
                    pa["diverged"] = s["diverged"]
                pa["epochs_run"] = int(s["epochs_run"])
                pa["ep_pos"] = int(s["ep_pos"])
                for k in ("probe_step0", "probe_floor", "step0_loss",
                          "noop_floor", "noop_floor_sd", "noop_floor_sem"):
                    if s.get(k) is not None:
                        pa[k] = s[k]
                if pa["stopper"] is not None and s.get("stopper") is not None:
                    st = s["stopper"]
                    pa["stopper"].best = st["best"]
                    pa["stopper"].best_epoch = st["best_epoch"]
                    pa["stopper"].epochs_seen = st["epochs_seen"]
                    pa["stopper"]._since_best = st["since_best"]
                    pa["stopper"]._snapshot = st["snapshot"]
            order_rng.setstate(ck["order_rng_state"])
            torch.set_rng_state(ck["torch_rng_state"])
            restore_cuda_rng(ck.get("cuda_rng_states"))
            step = int(ck["step"])
            start_epoch = int(ck["epoch"])
            resume_order = [int(b) for b in ck["order"]]
            resume_batches_done = int(ck["batches_done"])
            ckpt_last_steps = max(len(pa["history"]) for pa in per_arm)
            print(f"[fitckpt] train_lockstep RESUMING from checkpoint: "
                  f"epoch {start_epoch}, batch "
                  f"{resume_batches_done}/{n_batches} in epoch, optimizer "
                  f"steps so far {ckpt_last_steps} — skipping to the "
                  "checkpointed position", flush=True)

    for epoch in range(start_epoch, epochs):
        if all(pa["stopped"] for pa in per_arm):
            break
        if resume_order is not None and epoch == start_epoch:
            # Mid-epoch resume: the checkpoint carries this epoch's
            # already-shuffled order and how many batches were consumed
            # (per-arm ep_pos was restored above); the consumed batches'
            # optimizer steps live in the restored state, so they are
            # SKIPPED, not replayed.
            order = resume_order
            batches_done = resume_batches_done
        else:
            order = list(range(n_batches))
            order_rng.shuffle(order)
            batches_done = 0
            for pa in per_arm:
                pa["ep_pos"] = 0  # this arm's 0-based position in the epoch
        for bi in order[batches_done:]:
            # ONE teacher forward for this batch, shared by every arm. The
            # object is INACTIVE (no obj.apply), under no_grad — exactly the
            # solo teacher forward. Assert every arm's own teacher batch is
            # bit-identical to the shared one before trusting the sharing
            # (topk consumption included: the bound head stands for the
            # group only because the batches are identical).
            for pa in per_arm[1:]:
                if not _batches_equal(pa["teacher"][bi],
                                      per_arm[0]["teacher"][bi]):
                    raise ValueError(
                        f"teacher batch {bi} differs between {arms[0].name!r} "
                        f"and {pa['arm'].name!r} — the shared forward would be "
                        "wrong for at least one arm (completion truncation?)")
            if topk_batches is not None:
                # The precomputed head: no teacher forward at all — the
                # whole point of the offline precompute (frees the
                # [B, S, V] logits + [N, V] fp32 teacher tensors).
                t_idx, t_val = topk_batches[bi]
                shared_t_logits = (t_idx.to(device), t_val.to(device))
            else:
                shared_teacher = per_arm[0]["teacher"][bi].to(device)
                shared_t_logits = _train_teacher_logits(shared_teacher)

            for pa in per_arm:
                if pa["stopped"]:
                    continue
                a = pa["arm"]
                student_b = pa["student"][bi].to(device)
                # Restore this arm's RNG stream for its step (isolation).
                torch.set_rng_state(pa["rng_state"])
                # Accumulation-group bookkeeping (accum==1: one group per
                # batch, first == last == True, weights skipped — the
                # historical per-batch step, op for op).
                accum = pa["accum"]
                j = pa["ep_pos"]
                pa["ep_pos"] = j + 1
                g_start = (j // accum) * accum
                group = order[g_start:g_start + accum]
                first, last = (j == g_start), (j == g_start + len(group) - 1)
                w_div = w_reg = 1.0
                if accum > 1:
                    if first:
                        pa["acc_loss"] = 0.0
                        pa["acc_reg"] = None
                        pa["grp_tok"] = sum(pa["n_tok"][b] for b in group)
                        pa["grp_rows"] = sum(pa["n_rows"][b] for b in group)
                    w_div = pa["n_tok"][bi] / pa["grp_tok"]
                    w_reg = pa["n_rows"][bi] / pa["grp_rows"]
                # The arm's own ctx: a SteeringHandle-yielding (vector) arm
                # gets per-row boundaries + a mask-derived loss gather; a
                # mask-less arm gets the plain completion gather — the
                # mask→loss wiring, bit-identical for all-token arms.
                step_loss, reg = _masked_student_step(
                    model, a, student_b, shared_t_logits, pa["opt"],
                    zero_grad=first, w_div=w_div, w_reg=w_reg)
                if accum > 1:
                    pa["acc_loss"] += float(step_loss)
                    if reg is not None:
                        pa["acc_reg"] = ((pa["acc_reg"] or 0.0)
                                         + float(reg) * w_reg)
                    if not last:
                        # Mid-group: gradients accumulate; no clip, no
                        # optimizer step, no history/log/probe until the
                        # group completes.
                        pa["rng_state"] = torch.get_rng_state()
                        continue
                # Group end (every batch, at accum==1). ONE clip, on the
                # ACCUMULATED gradient — never per micro-batch, which
                # would silently reproduce micro-batch-level clipping
                # inside an effective-batch run. Per-arm schedule opt-in
                # (see LockstepArmParams): for a default arm BOTH branches
                # are dead code — no clip, no param_group write — so its
                # trajectory is bit-identical to the historical loop and
                # to its solo trainer.
                lr_t = None
                if (a.params.max_grad_norm is not None
                        and math.isfinite(a.params.max_grad_norm)):
                    torch.nn.utils.clip_grad_norm_(a.obj.trainable,
                                                   a.params.max_grad_norm)
                if pa["lr_at"] is not None:
                    lr_t = pa["lr_at"](len(pa["history"]))  # 0-based step
                    for g in pa["opt"].param_groups:
                        g["lr"] = lr_t
                pa["opt"].step()
                pa["rng_state"] = torch.get_rng_state()
                loss_val = (float(step_loss) if accum == 1
                            else pa["acc_loss"])
                reg_val = (None if reg is None else float(reg)) \
                    if accum == 1 else pa["acc_reg"]
                pa["history"].append(loss_val)
                if log_fn:
                    rec = {f"lockstep/{a.name}/{a.params.loss}_loss": loss_val,
                           "epoch": epoch}
                    if reg_val is not None:
                        rec[f"lockstep/{a.name}/gate_reg"] = reg_val
                    if lr_t is not None:
                        rec[f"lockstep/{a.name}/lr"] = lr_t
                    log_fn(rec, step)
                if probe_cb is not None:
                    # The production monitoring seam (see the docstring):
                    # grads are still present (post-clip) and the arm's
                    # optimizer step is done. A TrainingDiverged abort
                    # stops THIS arm only; the group trains on.
                    from .fitmon import TrainingDiverged

                    try:
                        probe_cb(step, a.name, a.obj, loss_val)
                    except TrainingDiverged as e:
                        pa["stopped"] = True
                        pa["diverged"] = str(e)
                        if log_fn:
                            log_fn({f"lockstep/{a.name}/diverged_step":
                                    step}, step)
                if pa["divguard"] is not None and not pa["stopped"]:
                    # Step-level divergence abort (docstring): rules on
                    # the loss float this step already synced; a verdict
                    # stops THIS ARM exactly like a probe_cb divergence —
                    # partners train on, no artifact ever ships.
                    try:
                        pa["divguard"].step(len(pa["history"]), loss_val)
                    except TrainingDiverged as e:
                        pa["stopped"] = True
                        pa["diverged"] = str(e)
                        print(f"[divergence-guard] {a.name}: {e}",
                              flush=True)
                        if log_fn:
                            log_fn({f"lockstep/{a.name}/diverged_step":
                                    step}, step)
            step += 1
            batches_done += 1
            if checkpoint_every is not None:
                # Fire only where EVERY live arm sits at an optimizer-step
                # boundary (its accumulation group just closed, or the
                # epoch's ragged tail did) — so a checkpoint never carries
                # partial-group state — and only once the group's
                # optimizer-step count has advanced a full interval since
                # the last write. At accum_steps=1 every batch is a
                # boundary and this is exactly every checkpoint_every
                # optimizer steps.
                live = [pa for pa in per_arm if not pa["stopped"]]
                at_boundary = all(
                    pa["ep_pos"] % pa["accum"] == 0
                    or pa["ep_pos"] == n_batches
                    for pa in live)
                opt_steps = max(len(pa["history"]) for pa in per_arm)
                if (live and at_boundary
                        and opt_steps - ckpt_last_steps >= checkpoint_every):
                    _save_ckpt(epoch, batches_done, order)
                    ckpt_last_steps = opt_steps
        for pa in per_arm:
            if not pa["stopped"]:
                pa["epochs_run"] = epoch + 1
        if heldout:
            _heldout_pass(epoch, step)
        if epoch_cb is not None:
            # EPOCH-BOUNDARY hook: after the epoch's held-out pass (so the caller can record this epoch's held-out
            # loss), with every arm's state — the caller saves a DURABLE copy of each live arm's object (epoch 1,
            # epoch 2 and the final epoch are the named artefacts; never rotated by the step-interval checkpointing
            # above, which stays as the resume mechanism). Called once per epoch for every arm, stopped arms
            # included (their object is the restored best-epoch one): the callback decides what to keep.
            epoch_cb(epoch, step, per_arm)

    # Arms that reached the cap while a better epoch sits in the patience
    # window keep the BEST epoch's weights, like the stopped arms.
    for pa in per_arm:
        if pa["stopper"] is not None and not pa["stopped"]:
            pa["stopper"].restore_best(pa["arm"].obj.trainable)

    # End-of-training probe: the trained (best-epoch) object on the SAME
    # probe batches as the startup check, against the SAME no-op floor —
    # the numbers distill.floor_ratio_check rules on (the caller enforces
    # it per arm so one degenerate arm cannot discard the group's work).
    # Teacher logits are recomputed per batch (the base is frozen, so they
    # are the startup batch's exact tensors) rather than held across the
    # whole run; the no-op leg is skipped (with_noop=False) because the
    # floor cannot have moved. The ratio's sem pairs each example's final
    # active value with ITS OWN startup floor, so the covariance term in
    # _ratio_sem is real, not assumed.
    for bi in range(n_probe_batches):
        probe_teacher = probe_teacher_batches[bi].to(device)
        # AUTHORITATIVE AND UNCACHED: this pass produces final_probe_loss —
        # the numerator floor_ratio_check rules on (rejection writes no
        # artifact, so this number must not carry cached-kernel drift).
        probe_t_logits = _authoritative_teacher_logits(probe_teacher)
        for pa in per_arm:
            if pa.get("diverged"):
                continue  # a diverged arm never ships; skip its probe
            fin, _noop = _probe_pair(model, pa["arm"], probe_t_logits,
                                     pa["probe_student"][bi].to(device),
                                     with_noop=False)
            pa.setdefault("probe_final", []).extend(fin)
    for pa in per_arm:
        a = pa["arm"]
        if pa.get("diverged"):
            # Aborted by the non-finite guard (probe_cb): record the
            # divergence so the caller can mark THIS arm failed — no final
            # probe, no floor ratio, never an artifact.
            if diagnostics is not None:
                diagnostics[a.name] = {
                    "loss": a.params.loss,
                    "diverged": pa["diverged"],
                    "step0_loss": pa["step0_loss"],
                    "noop_floor": pa["noop_floor"],
                    "final_probe_loss": None,
                    "floor_ratio": None,
                    "epochs_run": pa["epochs_run"],
                    "stopped_early": True,
                    "heldout_curve": pa["heldout_curve"],
                }
            continue
        final_probe, final_sd, final_sem = _mean_sd_sem(pa["probe_final"])
        pa["final_probe_loss"] = final_probe
        floor_ratio = final_probe / max(pa["noop_floor"], 1e-6)
        ratio_sem = _ratio_sem(pa["probe_final"], pa["probe_floor"])
        if log_fn:
            rec = {f"lockstep/{a.name}/final_probe_loss": final_probe,
                   f"lockstep/{a.name}/floor_ratio": floor_ratio}
            if final_sem is not None:
                rec[f"lockstep/{a.name}/final_probe_sem"] = final_sem
            if ratio_sem is not None:
                rec[f"lockstep/{a.name}/floor_ratio_sem"] = ratio_sem
            log_fn(rec, step)
        if diagnostics is not None:
            stopper = pa["stopper"]
            diagnostics[a.name] = {
                "loss": a.params.loss,
                "step0_loss": pa["step0_loss"],
                "noop_floor": pa["noop_floor"],
                "noop_floor_sd": pa["noop_floor_sd"],
                "noop_floor_sem": pa["noop_floor_sem"],
                "final_probe_loss": final_probe,
                "final_probe_sd": final_sd,
                "final_probe_sem": final_sem,
                "floor_ratio": floor_ratio,
                # The ratio's own uncertainty, recorded so a verdict (3.0x
                # today; the intended < 1.0 criterion) can be read WITH its
                # resolution rather than as a bare point value.
                "floor_ratio_sem": ratio_sem,
                # How many examples/batches the probe REALLY averaged over
                # (clamped to the pool), and WHICH examples: sha256-rank
                # selection over student renderings (see arm_probe_losses
                # — never a pool prefix, which is single-family), with the
                # selected keys' sha8 prefixes so the recorded floor names
                # its population.
                "probe_n": probe_n_realised,
                "probe_n_batches": n_probe_batches,
                "probe_selection": "sha256_rank(prompt_without)",
                "probe_selected_sha8": list(probe_sha8),
                "probe_batch_size": batch_size,
                # The per-example values themselves (2026-08-24): the final
                # active leg vs each example's OWN startup floor (the base
                # is frozen, so the floor could not move), both in probe-
                # selection order and aligned element-wise with
                # probe_selected_sha8 — what makes two cells probed on the
                # same examples paired-comparable from disk (correlated-
                # error sem, no GPU re-probe). per_family is None on this
                # path: the group probe is unstratified (train_lockstep
                # takes no family labels) — a stated absence.
                **probe_values_record(active_values=pa["probe_final"],
                                      noop_values=pa["probe_floor"],
                                      per_family=None),
                "epochs_run": pa["epochs_run"],
                "stopped_early": pa["stopped"],
                "heldout_curve": pa["heldout_curve"],
                "best_epoch": stopper.best_epoch if stopper else None,
                "best_heldout_loss": stopper.best if stopper else None,
                # The warmup this arm REALLY ramped over (None = no
                # schedule): a warmup_frac winner resolves against THIS
                # fit's budget, and the record must say to what.
                "resolved_warmup_steps": pa.get("resolved_warmup_steps"),
                # Gradient accumulation this arm REALLY trained under (1 =
                # the historical per-batch loop) and the optimizer-step
                # count it produced — effective batch = the shared
                # batch_size x accum_steps.
                "accum_steps": pa["accum"],
                "optimizer_steps": len(pa["history"]),
                # The teacher leg the TRAINING steps REALLY consumed (the
                # probes/held-out passes are always authoritative online
                # forwards): "precomputed_topk" only when a store was
                # passed AND bound — a row-binding fallback trains online
                # and must be recorded as online.
                "teacher_forward": ("precomputed_topk"
                                    if topk_batches is not None
                                    else "shared_lockstep"),
            }

    return {pa["arm"].name: pa["history"] for pa in per_arm}
