"""Bidirectional DPO (BiPO, Cao et al. 2406.00045) on lora_r1's substrate.

The matrix's DPO objective appears once, on the vector substrate (bidpo) —
so a bidpo-vs-lora_r1 gap conflates the OBJECTIVE (preference vs KL
distillation) with the SUBSTRATE (steering vector vs rank-1 LoRA).
``bidpo_lora`` is the de-confounding cell, completing the 2x2:

                 vector        LoRA r=1
    KL           vector        lora_r1
    DPO          bidpo         bidpo_lora   (this module)

``lora_r1`` vs ``bidpo_lora`` isolates the objective with the substrate
fixed; ``bidpo`` vs ``bidpo_lora`` isolates the substrate with the
objective fixed. Everything about the substrate therefore matches lora_r1
EXACTLY — the same :class:`~sisteer.extraction.objects.RankOneLoRAObject`
(dW = B A^T on the pinned layer's attention o_proj, B zero-init, the same
constructor call methods._train_lora makes) — and ONLY the training
objective differs (tests/test_bidpo_lora.py pins this).

Negation is well-defined here, which is what makes the substrate eligible
for a BIDIRECTIONAL preference objective at all: the low-rank edit is
additive (output += scale * B (A . x)), so ``scale=-1`` is exactly the
negated update -B A^T — a clean "-object". (A projection-based edit like
LoReFT's h + R^T(Wh + b - Rh) has no meaningful negation, so BiPO's second
term would be undefined there.)

Objective provenance — read before "fixing" any numeric here: the training
loop below is a deliberate line-for-line TRANSCRIPTION of
``extraction/bidpo.py``'s ``train_bidpo`` (BiPO Eq. 3: steering +object
must prefer chosen over rejected, -object must prefer rejected over
chosen, both against the unsteered frozen reference; same batching, same
reference computation, same margin and loss lines, same defaults), with
the trained vector generalized to any object exposing
``apply(model, scale)`` + ``trainable``. It could not IMPORT the loss:
train_bidpo holds it inline, and bidpo.py belongs to a peer branch's
training-loop rewrite, so factoring a shared helper out of it was off the
table. The drift guard is
tests/test_bidpo_lora.py::test_engine_matches_train_bidpo_on_the_vector_substrate,
which runs BOTH trainers on the same substrate/data/seed and requires
identical per-step losses and artifacts — if bidpo.py's objective ever
moves, that test fails loudly; re-align this engine (or factor the loss
into one shared helper once the peer branch lands) rather than loosening
the test. (A bidpo.py change to the vector SUBSTRATE's forward — e.g. the
2026-08-21 skip_bos=True deploy-parity fix — re-derives the test's
_VectorSubstrate adapter instead: the engine delegates the forward to the
object, so it has nothing to re-align, and the LoRA substrate's own
forward, which has no BOS notion and is train/deploy-consistent as is,
must not be bent to mirror a vector-only convention.)

Masked (decode-only) fits are WIRED (2026-08-21, user request — lora_r1
honors them and bidpo_lora is lora_r1's exact substrate, so the mask
machinery was already there and only this trainer's plumbing was
missing): ``positions="response"``
(+ ``response_header_len``) puts the object on its PositionMaskedFit leg,
so every policy forward's low-rank edit is active only at the handle's
positions (torch.where — masked positions bit-identical to the unsteered
model), and BOTH completion-logprob sums of each DPO margin term — the
policy's and the frozen reference's — gather exactly the completion
positions that mask covers (``loss_gather_masks`` +
``completion_logprobs(gather=...)``), so the margin compares like with
like. The gather comes from the SAME handle the hooks consumed
(train_object_kl's shape): the live handle apply() yields for the policy
forwards; for the reference — computed once, object INACTIVE, no live
handle — a transient vectorless tracker derives the identical mask, which
under per-row boundaries is a pure function of the batch
(train_object_kl's startup-probe pattern). The default ``positions="all"``
path is BIT-identical to the pre-wiring behavior (no handle is live, every
gather is None — the drift alarm above holds it to train_bidpo), and an
object exposing no ``set_fit_positions`` still refuses a masked fit loudly
(hooks.PositionMaskedFit's "an arm that silently pretends to be masked is
worse than one that refuses").
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from ..hooks import SteeringHandle
from .bidpo import PrefExample  # noqa: F401  (re-exported: the shared example type)
from .distill import loss_gather_masks, steer_prompt_boundaries
from .objects import RankOneLoRAObject
from .traindata import Batch, collate, completion_logprobs, encode_prompt_completion


def _live_handle(ctx_value):
    """The live SteeringHandle an object's ``apply()`` yielded, or ``None``
    — an object fitting all-token (RankOneLoRAObject's default leg yields
    None; the drift alarm's vector adapter yields nothing), i.e. the
    historical every-position convention (train_object_kl's helper)."""
    return ctx_value if isinstance(ctx_value, SteeringHandle) else None


def train_bidpo_object(
    model,
    tok,
    obj,
    examples: list[PrefExample],
    *,
    beta: float = 0.1,
    lr: float = 5e-4,
    weight_decay: float = 2.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    epochs: int = 4,
    batch_size: int = 2,
    accum_steps: int = 1,
    max_steps: int | None = None,
    lr_schedule: str | None = None,
    warmup_steps: int | None = None,
    max_len: int = 768,
    direction: str = "both",
    stop_pref_prob: float | None = None,
    positions: str = "all",
    response_header_len: int | None = None,
    seed: int = 0,
    log_fn=None,
    metric_prefix: str = "bidpo_lora",
) -> list[float]:
    """Train ``obj`` in place under train_bidpo's bidirectional DPO
    objective; returns the per-step loss history (train_object_kl's
    convention). Model weights are frozen throughout; the ONLY trainable
    state is ``obj.trainable``, applied via ``obj.apply(model, scale=d)``
    with d = +1 (the object) and d = -1 (its negation).

    Hyperparameter defaults are bidpo's (BiPO reference lr 5e-4, beta 0.1,
    completion-only loss; direction="both" computes the d=+1 and d=-1
    terms every step; weight_decay 2.0 — train_bidpo's 2026-08-24
    deviation from the reference's 0.05, whose rationale and wave
    evidence live on train_bidpo's docstring, the objective's home; kept
    IDENTICAL here because the recipe travels with the objective, and
    the old value stays reachable at weight_decay=0.05. Decay's target
    is the no-op on THIS substrate too: B is zero-init, dW = B A^T
    vanishes with it, and A = B = 0 — decay's fixed point — sits inside
    that no-op set) — deliberately NOT lora_r1's KL recipe: the
    optimizer recipe travels with the OBJECTIVE, so the bidpo-vs-bidpo_lora
    substrate comparison holds it fixed. batch_size defaults to 2, the
    width the same objective runs at on the vector substrate (see
    configs/two_si.yaml batch_size_by_arm) — NOT lora_r1's measured 8: a
    "both"-direction DPO step holds FOUR differentiable forward graphs
    (chosen/rejected x +-d) before its backward, where the KL step lora_r1
    was measured under holds one.

    ``stop_pref_prob`` — saturation-keyed early stopping, transcribed from
    train_bidpo (2026-08-24 wave post-mortem; the full rationale and the
    wave evidence live on train_bidpo's docstring, the objective's home):
    stop at the end of the first epoch whose MEAN training margin reaches
    logit(stop_pref_prob)/beta in BOTH directions — the Bradley-Terry
    preference the loss models is learned, and every further step only
    inflates the margin past saturation, growing the off-pool displacement
    floor_ratio_check refuses (the LoRA substrate saturates ~2x faster
    than the vector at every wave pool size, and was the arm that failed
    the floor at EVERY size, 9-52x — so this substrate needs the rule
    more, not less). Same semantics, same epoch-end event, same
    kept-weights convention; the drift alarm holds the two engines to
    identical histories under the rule. Default None keeps the historical
    fixed-budget behavior.

    ``positions``/``response_header_len`` select the fit's mask leg
    (hooks.FIT_POSITION_MODES): under ``"response"`` the object's hooks
    steer only the boundary-onward positions (header-steered when
    ``response_header_len`` is set — H is derived ONCE by the caller,
    never here) and BOTH completion-logprob sums of each margin term —
    policy and frozen reference — gather exactly the completion positions
    that mask covers, from the same mask the hooks consumed (see the
    module docstring). An object that exposes no fit mask
    (``set_fit_positions``) REFUSES the masked fit loudly: a silent
    all-token fit under a decode-only label is exactly the artifact the
    wiring exists to prevent. The default ``"all"`` path is bit-identical
    to the historical behavior.
    """
    if direction not in ("both", "sampled"):
        raise ValueError("direction must be 'both' or 'sampled'")
    if stop_pref_prob is not None and not (0.5 < stop_pref_prob < 1.0):
        # p <= 0.5 stops before anything is learned (margin <= 0); p = 1
        # is an infinite margin target — the unbounded push the rule
        # exists to cut short.
        raise ValueError(
            f"stop_pref_prob must lie in (0.5, 1), got {stop_pref_prob!r}")
    stop_margin = (math.log(stop_pref_prob / (1.0 - stop_pref_prob)) / beta
                   if stop_pref_prob is not None else None)
    # ``betas``/``eps``/``accum_steps``/``max_steps``/``lr_schedule``/
    # ``warmup_steps`` — transcribed from train_bidpo (the objective's home;
    # full rationale on its docstring): Adam's conditioning knobs into
    # AdamW; row-share-weighted gradient accumulation whose summed gradient
    # equals the true-batch gradient under mean reduction; a hard
    # optimizer-step budget; and the lockstep arms' warmup_cosine schedule
    # (one implementation — lockstep._lr_at_factory). Same names, same
    # semantics, same refusals, so the drift alarm's engine comparison
    # stays knob-for-knob meaningful.
    betas = tuple(float(b) for b in betas)
    if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
        raise ValueError(
            f"betas must be two floats in [0, 1), got {betas!r}")
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps!r}")
    accum_steps = int(accum_steps)
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps!r}")
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps!r}")
    if lr_schedule not in (None, "warmup_cosine", "constant"):
        raise ValueError(
            "lr_schedule must be None (fixed lr, the default), "
            f"'warmup_cosine' or 'constant' — got {lr_schedule!r}")
    if warmup_steps is not None and lr_schedule is None:
        raise ValueError(
            f"warmup_steps={warmup_steps} without an lr_schedule — refusing "
            "to silently ignore it (set lr_schedule='warmup_cosine' or "
            "'constant')")
    if positions != "all":
        set_fit = getattr(obj, "set_fit_positions", None)
        if set_fit is None:
            reason = getattr(
                obj, "FIT_MASK_REFUSAL",
                "its apply() exposes no fit position mask "
                "(no set_fit_positions)")
            raise ValueError(
                f"{type(obj).__name__} cannot honor a "
                f"positions={positions!r} (decode-only) fit: {reason} — "
                "never fit it all-token under a decode-only label.")
        set_fit(positions, response_header_len)
    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)

    for p in model.parameters():
        p.requires_grad_(False)
    obj.to_(device)
    params = list(obj.trainable)
    opt = torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay)

    def make_batches(field: str) -> list[Batch]:
        seqs = [
            encode_prompt_completion(tok, ex.prompt, getattr(ex, field), max_len)
            for ex in examples
        ]
        return [
            collate(seqs[i : i + batch_size], tok.pad_token_id)
            for i in range(0, len(seqs), batch_size)
        ]

    chosen_batches = make_batches("chosen")
    rejected_batches = make_batches("rejected")

    # The planned optimizer-step budget: the schedule's horizon (an early
    # stop may end sooner — lockstep's convention, the horizon is the cap).
    steps_per_epoch = math.ceil(len(chosen_batches) / accum_steps)
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    lr_at = None
    if lr_schedule is not None:
        from .lockstep import _lr_at_factory

        warmup_resolved = (int(warmup_steps) if warmup_steps is not None
                           else min(500, max(1, total_steps // 10)))
        lr_at = _lr_at_factory(lr, lr_schedule, warmup_resolved, total_steps)

    def _ref_fit_gather(batch: Batch) -> torch.Tensor:
        """The mask→loss gather for one reference batch, derived WITHOUT a
        live forward (train_object_kl's startup-probe pattern): the
        reference runs with the object INACTIVE, so no live handle exists —
        and the handle's mask under per-row boundaries is a pure function
        of the batch, so a transient vectorless tracker yields the same
        gather the policy forwards derive from their live handle."""
        h = obj._fit_mask_handle(model)
        try:
            h.set_prompt_lens(
                steer_prompt_boundaries(batch, h.response_header_len))
            gather, _t_gather, _keep = loss_gather_masks(h, batch)
        finally:
            h.remove()
        return gather

    # Reference logprobs: the object INACTIVE (no apply), computed once —
    # the same frozen unsteered reference train_bidpo uses. Under a masked
    # fit each sum is restricted to the SAME completion∩mask positions the
    # policy's sums read, so the margin compares like with like; on the
    # default leg every gather is None and the sums are bit-identical to
    # the historical ones.
    ref_c, ref_r = [], []
    with torch.no_grad():
        for bc, br in zip(chosen_batches, rejected_batches):
            bc, br = bc.to(device), br.to(device)
            gc = _ref_fit_gather(bc) if positions != "all" else None
            gr = _ref_fit_gather(br) if positions != "all" else None
            ref_c.append(completion_logprobs(model(
                input_ids=bc.input_ids, attention_mask=bc.attention_mask
            ).logits, bc, gather=gc))
            ref_r.append(completion_logprobs(model(
                input_ids=br.input_ids, attention_mask=br.attention_mask
            ).logits, br, gather=gr))

    def steered_logprobs(batch: Batch, sign_coef: float) -> torch.Tensor:
        # The mask→loss wiring (train_object_kl's shape): a masked object's
        # apply() yields its live mask handle — set per-row boundaries
        # BEFORE the forward (never derived from the padded batch width)
        # and gather exactly the completion positions the hook's mask
        # covers. handle is None on the default all-token leg, which stays
        # bit-identical (gather None -> the historical completion sum).
        gather = None
        with obj.apply(model, scale=sign_coef) as ctx_value:
            handle = _live_handle(ctx_value)
            if handle is not None and handle.positions in (
                    "response", "prompt"):
                handle.set_prompt_lens(steer_prompt_boundaries(
                    batch, handle.response_header_len))
            logits = model(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).logits
            if handle is not None:
                gather, _t_gather, _keep = loss_gather_masks(handle, batch)
        return completion_logprobs(logits, batch, gather=gather)

    history: list[float] = []
    step = 0
    budget_exhausted = False
    order = list(range(len(chosen_batches)))
    for epoch in range(epochs):
        rng.shuffle(order)
        # Per-direction epoch-mean training margins, the stop rule's
        # evidence (only tracked when the rule is armed).
        epoch_margins = ({+1.0: [], -1.0: []} if stop_margin is not None
                         else None)
        for g0 in range(0, len(order), accum_steps):
            group = order[g0:g0 + accum_steps]
            # One direction draw per OPTIMIZER step (train_bidpo's
            # accumulation semantics: the true batch this group
            # accumulates draws d once). At accum_steps=1 this is the
            # historical per-batch draw — same rng stream, bit-identical.
            d_sampled = rng.choice([-1.0, 1.0]) if direction == "sampled" \
                else None
            # Row-share weighting, transcribed from train_bidpo: the
            # summed gradient equals the true-batch gradient under mean
            # reduction, ragged tail included; w == 1.0 skips the multiply
            # so the single-batch step stays bit-identical.
            n_group = sum(int(chosen_batches[bi].input_ids.shape[0])
                          for bi in group)
            loss_val = 0.0
            fwd_mean = rev_mean = step_mean = None
            opt.zero_grad()
            for bi in group:
                bc = chosen_batches[bi].to(device)
                br = rejected_batches[bi].to(device)
                w = int(bc.input_ids.shape[0]) / n_group

                def margin(d: float, bc=bc, br=br, bi=bi) -> torch.Tensor:
                    # d * beta * [(pol(chosen)-ref(chosen)) - (pol(rejected)-ref(rejected))]
                    # with the policy's object applied at scale d (BiPO Eq. 3);
                    # d=-1 applies the NEGATED low-rank update.
                    pol_c = steered_logprobs(bc, d)
                    pol_r = steered_logprobs(br, d)
                    return d * ((pol_c - ref_c[bi]) - (pol_r - ref_r[bi]))

                if direction == "sampled":
                    step_margin = margin(d_sampled)
                    loss = -F.logsigmoid(beta * step_margin).mean()
                    fwd_margin = rev_margin = None
                    step_mean = (step_mean or 0.0) + w * float(
                        step_margin.detach().mean())
                else:
                    fwd_margin = margin(+1.0)
                    rev_margin = margin(-1.0)
                    loss = -(
                        F.logsigmoid(beta * fwd_margin)
                        + F.logsigmoid(beta * rev_margin)
                    ).mean()
                    fwd_mean = (fwd_mean or 0.0) + w * float(
                        fwd_margin.detach().mean())
                    rev_mean = (rev_mean or 0.0) + w * float(
                        rev_margin.detach().mean())
                (loss if w == 1.0 else loss * w).backward()
                loss_val += w * float(loss.detach())
            if epoch_margins is not None:
                if direction == "sampled":
                    epoch_margins[d_sampled].append(step_mean)
                else:
                    epoch_margins[+1.0].append(fwd_mean)
                    epoch_margins[-1.0].append(rev_mean)
            # Per-OPTIMIZER-step global grad norm over the trainable LoRA
            # params (train_bidpo's telemetry, transcribed): clip_grad_norm_
            # at max_norm=inf returns exactly the total norm and clips
            # nothing (clip coef clamps to 1.0). Never per micro-batch.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                params, float("inf")))
            if lr_at is not None:
                lr_now = lr_at(step)
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
            opt.step()
            step += 1
            history.append(loss_val)
            if log_fn:
                metrics = {
                    f"{metric_prefix}/loss": loss_val,
                    f"{metric_prefix}/param_norm": float(torch.sqrt(sum(
                        p.detach().float().norm() ** 2 for p in params))),
                    "epoch": epoch,
                }
                if lr_at is not None:
                    metrics[f"{metric_prefix}/lr"] = lr_now
                metrics[f"{metric_prefix}/grad_norm"] = grad_norm
                if fwd_mean is not None:
                    metrics[f"{metric_prefix}/fwd_margin"] = fwd_mean
                    metrics[f"{metric_prefix}/rev_margin"] = rev_mean
                log_fn(metrics, step)
            if max_steps is not None and step >= max_steps:
                # Hard step budget: stop mid-epoch, skip the stop-rule
                # check (this epoch's margin evidence is partial).
                budget_exhausted = True
                break
        if budget_exhausted:
            break
        if epoch_margins is not None:
            means = {d: sum(v) / len(v)
                     for d, v in epoch_margins.items() if v}
            if len(means) == 2 and all(m >= stop_margin
                                       for m in means.values()):
                if log_fn:
                    log_fn({f"{metric_prefix}/early_stop_epoch": epoch,
                            f"{metric_prefix}/early_stop_margin_fwd":
                                means[+1.0],
                            f"{metric_prefix}/early_stop_margin_rev":
                                means[-1.0]},
                           step)
                break

    return history


def train_bidpo_lora(
    model,
    tok,
    examples: list[PrefExample],
    layer: int,
    *,
    beta: float = 0.1,
    lr: float = 5e-4,
    weight_decay: float = 2.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    epochs: int = 4,
    batch_size: int = 2,
    accum_steps: int = 1,
    max_steps: int | None = None,
    lr_schedule: str | None = None,
    warmup_steps: int | None = None,
    max_len: int = 768,
    direction: str = "both",
    stop_pref_prob: float | None = None,
    positions: str = "all",
    response_header_len: int | None = None,
    seed: int = 0,
    log_fn=None,
) -> RankOneLoRAObject:
    """The bidpo_lora arm: bidpo's objective on lora_r1's exact substrate.

    The object construction is VERBATIM methods._train_lora's
    (``RankOneLoRAObject(model, layer=layer)`` — same rank-1 dW = B A^T on
    the pinned layer's o_proj, same seed-0 A init, same zero B so step 0 is
    exactly the unsteered model, which here doubles as the DPO reference).
    Anything added or changed here beyond the objective would re-confound
    the lora_r1 comparison this arm exists for. Returns the trained object
    — the SAME artifact type lora_r1 produces (kind "lora", object.pt),
    hydrating through the existing _KIND_LOADERS entry.

    ``positions``/``response_header_len`` select the fit's mask leg — the
    engine honors it through the object's own PositionMaskedFit machinery
    (see train_bidpo_object); the driver passes both as PROTECTED kwargs,
    so a masked-fit request reaches this signature or fails loudly.
    """
    obj = RankOneLoRAObject(model, layer=layer)
    train_bidpo_object(
        model, tok, obj, examples,
        beta=beta, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
        epochs=epochs, batch_size=batch_size, accum_steps=accum_steps,
        max_steps=max_steps, lr_schedule=lr_schedule,
        warmup_steps=warmup_steps, max_len=max_len, direction=direction,
        stop_pref_prob=stop_pref_prob,
        positions=positions, response_header_len=response_header_len,
        seed=seed, log_fn=log_fn,
    )
    return obj
