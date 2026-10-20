"""BiDPO / BiPO (Cao et al. 2406.00045): preference-optimized steering vector.

Trains a single d_model vector v at one layer with a bi-directional DPO
objective: steering +v must make the model prefer `chosen` over
`rejected`, and steering -v must prefer `rejected` over `chosen`, both
relative to the unsteered reference model (same weights, no vector).

For system-instruction distillation the pairs are, per user prompt:
chosen = completion generated WITH the instruction in context,
rejected = completion generated WITHOUT (or with the anti-instruction).
The prompt seen during training contains NO instruction — the vector has
to carry it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..hooks import FIT_POSITION_MODES, SteeringHandle, steering
from ..models import hidden_size
from ..vectors import SteeringVector
from .distill import loss_gather_masks, steer_prompt_boundaries
from .traindata import Batch, collate, completion_logprobs, encode_prompt_completion


class BidpoNonFiniteError(RuntimeError):
    """A NaN/inf optimizer-step loss or gradient norm: the fit diverged.

    Raised BEFORE the poisoned step is applied — ``train_bidpo`` must never
    return a vector trained past a non-finite value. The caller records the
    fit as failed (the sweep's divergence outcome) instead of shipping an
    artifact whose weights passed through a NaN.
    """


@dataclass
class PrefExample:
    prompt: str  # rendered WITHOUT the instruction
    chosen: str
    rejected: str


def train_bidpo(
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
    batch_size: int = 4,
    accum_steps: int = 1,
    max_steps: int | None = None,
    max_grad_norm: float | None = None,
    lr_schedule: str | None = None,
    warmup_steps: int | None = None,
    warmup_frac: float | None = None,
    max_len: int = 768,
    direction: str = "both",
    stop_pref_prob: float | None = None,
    positions: str = "all",
    response_header_len: int | None = None,
    skip_bos: bool = True,
    instruction_id: str = "",
    seed: int = 0,
    log_fn=None,
    log_every: int = 25,
    probe_cb: Callable[[int, object], None] | None = None,
    probe_every: int | None = None,
) -> SteeringVector:
    """Returns the trained vector. Model weights are frozen throughout.

    ``betas``/``eps`` — Adam's own conditioning knobs, passed straight to
    AdamW (2026-08-25: they were silently pinned at the torch defaults
    (0.9, 0.999)/1e-8 before, which mattered because Adam's scale
    invariance is the engine of the post-saturation norm march the
    weight-decay note below describes — past saturation the raw gradient
    collapses as sigma(-beta*margin) while m_hat/sqrt(v_hat) stays O(1).
    eps breaks that invariance directly (once sqrt(v_hat) << eps the
    update is ~lr*m_hat/eps, shrinking with the gradient again); beta2
    closer to 1 keeps v_hat remembering large early gradients while m_hat
    decays at beta1's faster rate). Recorded in the artifact metadata.

    ``accum_steps`` — gradient accumulation: each optimizer step spans
    ``accum_steps`` consecutive micro-batches of ``batch_size`` rows, every
    micro-batch's mean loss weighted by its ROW SHARE of the group (never a
    flat 1/accum_steps, which silently biases the mean whenever the last
    group is ragged), so the accumulated gradient EQUALS the gradient of
    one true batch of those rows under mean reduction — the DPO
    effective-batch floor (>=128 target, 32 hard floor) on hardware that
    cannot hold the four differentiable forward graphs of a true 128-row
    "both" step. Under ``direction="sampled"`` the sign d is drawn once per
    OPTIMIZER step (a true batch draws once). ``accum_steps=1`` is
    bit-identical to the historical loop. (This branch runs the DPO arms
    single-process — no row-sharded fit context exists here.)

    ``max_steps`` — hard optimizer-step budget: training stops mid-epoch
    when reached (``steps_run``/``epochs_run`` in metadata say what really
    ran). The fair-comparison unit for a sweep whose trials share a step
    budget rather than an epoch count.

    ``max_grad_norm`` — global-norm clip on the vec gradient, applied AFTER
    the telemetry norm is read and BEFORE ``opt.step()``: ``bidpo/grad_norm``
    is therefore always the PRE-clip norm (comparable across clip settings,
    and the divergence signal a monitor watches) and
    ``bidpo/grad_norm_clipped`` is what the optimizer actually consumed.
    This is the clip axis of the lr x clip sweep — without it the sweep's
    second axis would never reach the optimizer. None (default) keeps the
    historical unclipped step; ``final_grad_norm``/``max_grad_norm_seen``
    metadata stay pre-clip either way.

    ``lr_schedule``/``warmup_steps`` — the lockstep arms' per-arm optimiser
    schedule, same vocabulary and same formula (lockstep._lr_at_factory:
    linear ramp ``lr*(i+1)/warmup_steps`` then cosine decay to 0 over the
    FULL planned budget — min(epochs*steps_per_epoch, max_steps); an early
    stop may end sooner, the horizon is the cap). None = the historical
    fixed-lr loop, bit-identical; ``warmup_steps`` without a schedule is
    refused rather than silently ignored (lockstep's rule). When
    scheduled, the live lr is logged per step as ``bidpo/lr``.

    ``warmup_frac`` — fractional warmup, mutually exclusive with
    ``warmup_steps``: resolved against THIS run's own planned optimizer-step
    horizon (min(epochs*steps_per_epoch, max_steps)) as
    ``max(1, round(warmup_frac * total_steps))``, so the same configured
    fraction means the same schedule SHAPE across pools and budgets whose
    step counts differ. Refused without a schedule, exactly like
    ``warmup_steps``; the resolved count lands in metadata
    ``warmup_steps`` alongside the fraction itself.

    ``probe_cb``/``probe_every`` — the trainer-level probe seam: the live
    vector is internal to this function, so mid-fit probes (dual-KL
    monitoring, the tune adapter's mid-fit KL) get ``probe_cb(step, obj)``
    invoked right after the optimizer step at every step where
    ``step % probe_every == 0`` (1-based optimizer steps). ``obj`` is a
    TRANSIENT ``VectorObject`` wrapping a detached CLONE of the current
    vec under the fit's own deploy convention (same positions/skip_bos/
    response_header_len) — safe to forward under ``obj.apply(model)``,
    never aliasing the live parameter, never persisted. The two parameters
    come together: a callback with no cadence (or a cadence with no
    callback) is refused rather than left silently inert.

    ``log_every`` — the telemetry flush cadence (fitmon.ProbeCadence's
    ``log_every``, same default). Per-step scalars — loss, margins, grad/
    vec norms — are accumulated as 0-dim DEVICE tensors and materialised
    in ONE stacked device->host transfer every ``log_every`` optimizer
    steps (also at the probe cadence and at end of fit), at which point
    ``log_fn`` receives the buffered records, one per optimizer step, in
    step order. Content is unchanged from the historical per-step call;
    only the emission is batched: a ``float()`` per scalar per micro-batch
    forces a cudaStreamSynchronize each — ~96+ per optimizer step at
    accum_steps=32 under direction="both" — which destroys the ~100% GPU
    duty the large fits depend on (the gpu-util sync discipline; fitmon's
    module docstring is the reference implementation). The non-finite
    guard below still rules per optimizer step BEFORE ``opt.step()``, via
    a single pre-reduced device bool — the one remaining per-step sync.

    Non-finite guard: a NaN/inf optimizer-step loss or gradient norm
    aborts the fit with ``BidpoNonFiniteError`` BEFORE the poisoned step
    is applied — a diverged fit fails loudly and returns nothing; the
    sweep records the trial as failed and moves on.

    Reference (CaoYuanpu/BiPO): d ~ Uniform{-1,+1} sampled per batch, applied
    both inside the forward (+d*v) and as an outer sign on the DPO logits;
    lr 5e-4, wd 0.05, beta 0.1, zero init, completion-only loss, vector added
    at every token position of one layer. direction="sampled" reproduces
    that; "both" computes the d=+1 and d=-1 terms every step (same
    expectation, lower variance, 2x compute).

    Deviation ONE from the reference (2026-08-21): ``skip_bos``
    defaults True, so the training forward runs under the artifact's DEPLOY
    convention — methods.load_object serves every plain (non-antidote)
    SteeringVector as a ``VectorObject``, whose default is skip_bos=True
    (HyperSteer's convention; every other vector trainer already fits this
    way: distill/ladder plumb the flag, lockstep's vector arms hardcode it).
    Before this, bidpo alone optimised its vector while perturbing the BOS
    attention sink — very high norm in Llama-3 — and then deployed where
    that perturbation is absent. Pass ``skip_bos=False`` for the
    paper-literal forward.

    Deviation TWO (2026-08-24 wave post-mortem — the same evidence behind
    ``stop_pref_prob``): ``weight_decay`` defaults 2.0, not the reference's
    0.05. The knob was always explicit here and plumbed into AdamW; the
    change is the VALUE, and the old recipe stays reachable bitwise at
    ``weight_decay=0.05``. Why: AdamW's decoupled decay contracts the
    vector by lr*wd per step, so at lr 5e-4 the old 0.05 has a time
    constant of 1/(lr*wd) = 40,000 steps — ~35x the wave's 6-epoch runs
    (~1,150-2,000 steps), a ~3% total pull, inert. Those runs optimized
    the loss monotonically (0.577 -> 0.123, epoch means) while
    bidpo/vec_norm grew 1.11 -> 3.44 with NO plateau and the artifact
    failed floor_ratio_check at 14.3x the no-op floor. The growth
    mechanism is Adam normalization: past saturation the raw gradient
    collapses (sigma(-beta*margin)) but m/sqrt(v) stays O(1), so the norm
    keeps marching at near-constant velocity — measured ~2.4e-3/step, an
    effective push of ~4.8 norm-units per unit lr. Decay balances that
    push where wd*||v|| = push, i.e. ||v||* ~ push/wd: ~96 at 0.05 (never
    reached, hence no plateau), ~2.4 at 2.0 — the norm band the
    floor-passing runs ended in — with the time constant (1,000 steps)
    inside every wave run's budget. The brake also points at the right
    target: this vector's no-op is EXACTLY v = 0 (zero init; the
    unsteered model is the DPO reference), so decay pulls toward "do
    nothing" — unlike loreft, whose no-op is the nonzero rotation R and
    which pins wd=0 for precisely that reason (reft.py). And it barely
    touches learning: at the wave's norms the per-coordinate pull is ~11%
    of a full-signal Adam step, so decay only wins once the gradient
    signal has collapsed — the inflation phase it exists to stop. It
    complements ``stop_pref_prob`` rather than duplicating it: the stop
    cuts the budget at the saturation EVENT; decay bounds the norm
    continuously, including when a margin never clears the stop's bar in
    both directions (where the stop cannot fire at all).

    ``stop_pref_prob`` — saturation-keyed early stopping (2026-08-24 wave
    post-mortem). The DPO loss models the preference as the Bradley-Terry
    probability sigma(beta * margin); -logsigmoid keeps a nonzero gradient
    at EVERY margin, so on a preference pool small enough to saturate
    before the epoch budget runs out, the optimizer spends the remaining
    epochs inflating an already-won margin — a bigger, narrower
    discrimination whose off-pool displacement is exactly what
    floor_ratio_check refuses. Measured on the 6-epoch wave (Llama-3.1-8B,
    per-SI pools): the small pools (766/872 harmful pairs) crossed
    epoch-mean beta*margin ~3 by epoch 1-2 and ended at ~6 (both classes,
    floor 14-15x, FAILED); the one large pool (1324) only REACHED
    beta*margin 3.0 ≈ p 0.95 at the final epoch and PASSED at 2.7x — so
    the failure tracks time-spent-past-saturation, not pool size per se.
    The rule: stop at the end of the first epoch whose MEAN training
    margin reaches logit(stop_pref_prob)/beta in BOTH directions (each
    example votes exactly once per epoch — an unbiased pool estimate, no
    extra forwards; under direction="sampled" a direction not sampled this
    epoch cannot clear the bar). The weights kept are the stopping epoch's:
    the criterion is "the preference is learned", not a held-out best to
    restore. Replayed against the wave's logged margins, p=0.95 leaves the
    passing runs untouched (they stop at the final epoch) and cuts every
    failing run at epoch 1-2, inside the margin band the passing objects
    ended in. Default None keeps the historical fixed-budget behavior.

    ``positions``/``response_header_len`` select the fit's mask leg
    (hooks.FIT_POSITION_MODES — the same wiring as bidpo_lora's engine,
    train_object_kl's shape): under ``"response"`` every policy forward sets
    per-row boundaries on its live steering handle BEFORE the forward
    (steer_prompt_boundaries — never the padded batch width; header-steered
    when ``response_header_len`` is set, H derived ONCE by the caller), and
    BOTH completion-logprob sums of each DPO margin term — the policy's and
    the frozen reference's — gather exactly the completion positions that
    mask covers (``loss_gather_masks`` + ``completion_logprobs(gather=...)``),
    so the margin compares like with like. The reference, computed with no
    vector applied, derives the identical mask from a transient vectorless
    tracker (PositionMaskedFit's startup-probe pattern — the mask under
    per-row boundaries is a pure function of the batch). The default
    ``"all"`` leg keeps the historical loss support (every gather is None).
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
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
        if max_grad_norm <= 0.0:
            raise ValueError(
                f"max_grad_norm must be positive, got {max_grad_norm!r} "
                "(None disables clipping)")
    if lr_schedule not in (None, "warmup_cosine", "constant"):
        raise ValueError(
            "lr_schedule must be None (fixed lr, the default), "
            f"'warmup_cosine' or 'constant' — got {lr_schedule!r}")
    if warmup_steps is not None and warmup_frac is not None:
        raise ValueError(
            f"warmup_steps={warmup_steps} and warmup_frac={warmup_frac} are "
            "mutually exclusive — set one (warmup_frac resolves against "
            "this run's own planned optimizer steps)")
    if warmup_frac is not None:
        warmup_frac = float(warmup_frac)
        if not (0.0 < warmup_frac < 1.0):
            raise ValueError(
                f"warmup_frac must lie in (0, 1), got {warmup_frac!r}")
    if warmup_steps is not None and lr_schedule is None:
        raise ValueError(
            f"warmup_steps={warmup_steps} without an lr_schedule — refusing "
            "to silently ignore it (set lr_schedule='warmup_cosine' or "
            "'constant')")
    if warmup_frac is not None and lr_schedule is None:
        raise ValueError(
            f"warmup_frac={warmup_frac} without an lr_schedule — refusing "
            "to silently ignore it (set lr_schedule='warmup_cosine' or "
            "'constant')")
    if (probe_cb is None) != (probe_every is None):
        raise ValueError(
            "probe_cb and probe_every come together — a callback with no "
            "cadence (or a cadence with no callback) is silently inert; got "
            f"probe_cb={'set' if probe_cb is not None else None}, "
            f"probe_every={probe_every!r}")
    if probe_every is not None:
        probe_every = int(probe_every)
        if probe_every < 1:
            raise ValueError(
                f"probe_every must be >= 1, got {probe_every!r}")
    log_every = int(log_every)
    if log_every < 1:
        raise ValueError(f"log_every must be >= 1, got {log_every!r}")
    if positions not in FIT_POSITION_MODES:
        # The fit vocabulary (the train half of the train x test 2x2), the
        # same refusal set_fit_positions gives the object arms: "prompt" /
        # "front" / "last" are serving modes, not fit legs, and before the
        # 2026-08-21 wiring any non-"all" value here was silently INERT
        # (no boundary ever set -> "response" meant all positions).
        raise ValueError(
            f"fit positions must be one of {FIT_POSITION_MODES}, got "
            f"{positions!r}")
    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)

    for p in model.parameters():
        p.requires_grad_(False)
    vec = nn.Parameter(torch.zeros(hidden_size(model), device=device))
    opt = torch.optim.AdamW([vec], lr=lr, betas=betas, eps=eps,
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
    warmup_resolved = None
    lr_at = None
    if lr_schedule is not None:
        from .lockstep import _lr_at_factory, resolve_warmup_steps

        if warmup_steps is not None:
            warmup_resolved = int(warmup_steps)
        elif warmup_frac is not None:
            # Against THIS run's own horizon — the max_steps-capped budget,
            # never the raw epoch count — so a capped trial warms up over
            # the steps it will actually take.
            warmup_resolved = resolve_warmup_steps(warmup_frac, total_steps)
        else:
            warmup_resolved = min(500, max(1, total_steps // 10))
        lr_at = _lr_at_factory(lr, lr_schedule, warmup_resolved, total_steps)

    def _live_probe_object(at_step: int):
        # The probe seam's transient object: a detached CLONE of the current
        # vec behind the object protocol, under the fit's own deploy
        # convention. A clone, not a view — the probe may outlive this
        # optimizer step, and the live parameter is not its to watch.
        from ..methods import VectorObject  # lazy: methods imports this module

        live = SteeringVector(
            vectors={layer: vec.detach().clone()},
            method="bidpo",
            model_name=model.config._name_or_path,
            instruction_id=instruction_id,
            metadata={"transient_probe": True, "step": at_step},
        )
        return VectorObject(live, positions=positions, skip_bos=skip_bos,
                            response_header_len=response_header_len)

    def _ref_fit_gather(batch: Batch) -> torch.Tensor:
        """The mask→loss gather for one reference batch, derived WITHOUT a
        live forward (bidpo_lora's `_ref_fit_gather` / train_object_kl's
        startup-probe pattern): the reference runs unsteered, so no live
        handle exists — and the handle's mask under per-row boundaries is a
        pure function of the batch, so a transient vectorless tracker
        (carrying the SAME flags as the policy forwards' handles) yields the
        same gather the policy forwards derive from their live handle."""
        h = SteeringHandle(model, {}, positions=positions, skip_bos=skip_bos,
                           response_header_len=response_header_len)
        try:
            h.set_prompt_lens(
                steer_prompt_boundaries(batch, h.response_header_len))
            gather, _t_gather, _keep = loss_gather_masks(h, batch)
        finally:
            h.remove()
        return gather

    # Reference logprobs: unsteered model, computed once. Under a masked fit
    # each sum is restricted to the SAME completion∩mask positions the
    # policy's sums read, so the margin compares like with like; on the
    # default leg every gather is None and the sums are the historical ones.
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
        # The mask→loss wiring (bidpo_lora's shape): under a masked fit, set
        # per-row boundaries on the live handle BEFORE the forward (never
        # derived from the padded batch width) and gather exactly the
        # completion positions the hook's mask covers. On the default "all"
        # leg the gather stays None — the historical completion sum.
        gather = None
        with steering(model, {layer: vec}, scale=sign_coef,
                      positions=positions, skip_bos=skip_bos,
                      response_header_len=response_header_len) as handle:
            if positions != "all":
                handle.set_prompt_lens(
                    steer_prompt_boundaries(batch, response_header_len))
            logits = model(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            ).logits
            if positions != "all":
                gather, _t_gather, _keep = loss_gather_masks(handle, batch)
        return completion_logprobs(logits, batch, gather=gather)

    history: list[float] = []
    step = 0
    epochs_run = 0
    budget_exhausted = False
    # Gradient-norm telemetry (2026-08-25): with the floor probe existing
    # only at end of fit, the signals a monitor can act on mid-run are the
    # ones intrinsic to the optimisation — loss, grad norm, vec norm. The
    # grad norm is per OPTIMIZER step (after all micro-batches' backwards),
    # never per micro-batch — a micro-batch norm is a different quantity
    # and incomparable across accum_steps settings. Always PRE-clip, so the
    # series stays comparable across max_grad_norm settings.
    last_grad_norm = max_grad_seen = None
    # Sync discipline (gpu-util skill; fitmon's module docstring): every
    # per-step scalar stays a 0-dim DEVICE tensor — a float()/.item() per
    # metric forces a cudaStreamSynchronize each, and the historical
    # per-micro-batch float()s cost ~3 syncs x accum_steps per optimizer
    # step, which tanks the GPU duty the 27B fits depend on. Records
    # buffer in `pending` and materialise in ONE stacked D2H transfer per
    # flush (every log_every steps, at the probe cadence, at end of fit,
    # and before the non-finite abort's raise — FitMonitor.flush's shape).
    # The one remaining per-step sync is the non-finite guard's
    # pre-reduced bool, which must be read BEFORE opt.step().
    pending: list[tuple[int, dict]] = []

    def _flush() -> None:
        """Materialise the buffered records (tensor -> float in ONE
        stacked device transfer for the whole backlog — a per-value
        float() here would reintroduce the per-scalar round-trips), fold
        the loss / grad-norm series into history and the final/max
        bookkeeping, and emit each record through log_fn in step order."""
        nonlocal last_grad_norm, max_grad_seen
        if not pending:
            return
        buffered = pending[:]
        pending.clear()
        tensors = [v for _s, rec in buffered for v in rec.values()
                   if isinstance(v, torch.Tensor)]
        flat = iter(torch.stack([t.reshape(()).double() for t in tensors])
                    .tolist() if tensors else ())  # the ONE D2H sync
        for at_step, rec in buffered:
            vals = {k: (next(flat) if isinstance(v, torch.Tensor) else v)
                    for k, v in rec.items()}
            if "bidpo/loss" in vals:
                history.append(vals["bidpo/loss"])
            gn = vals.get("bidpo/grad_norm")
            if gn is not None:
                last_grad_norm = gn
                max_grad_seen = (gn if max_grad_seen is None
                                 else max(max_grad_seen, gn))
            if log_fn:
                log_fn(vals, at_step)

    order = list(range(len(chosen_batches)))
    for epoch in range(epochs):
        rng.shuffle(order)
        # Per-direction epoch-mean training margins, the stop rule's
        # evidence (only tracked when the rule is armed).
        epoch_margins = ({+1.0: [], -1.0: []} if stop_margin is not None
                         else None)
        for g0 in range(0, len(order), accum_steps):
            group = order[g0:g0 + accum_steps]
            # One direction draw per OPTIMIZER step: the true batch this
            # group accumulates draws d once, so accumulation keeps the
            # sampled-direction semantics of the equivalent large batch.
            # At accum_steps=1 this is the historical per-batch draw —
            # same rng stream, bit-identical.
            d_sampled = rng.choice([-1.0, 1.0]) if direction == "sampled" \
                else None
            opt.zero_grad()
            # Gradient accumulation: each micro-batch's mean loss is
            # weighted by its ROW SHARE of the group, so the summed
            # gradient equals the gradient of one true batch of these
            # rows under mean reduction (a flat 1/accum_steps would
            # silently bias the mean whenever the last group is
            # ragged). Logged loss/margins are the same row-weighted
            # means the true batch's .mean() would report. w == 1.0
            # (accum_steps=1) skips the multiply: bit-identical to the
            # historical single-batch step.
            n_group = sum(int(chosen_batches[bi].input_ids.shape[0])
                          for bi in group)
            # 0-dim DEVICE accumulators, float64 so the accumulation is
            # bit-identical to the historical `w * float(...)` Python-float
            # sums (fp32 -> fp64 is exact; same op order) — never a float()
            # per micro-batch (the sync discipline above).
            loss_t = None
            fwd_mean = rev_mean = step_mean = None
            for bi in group:
                bc = chosen_batches[bi].to(device)
                br = rejected_batches[bi].to(device)
                w = int(bc.input_ids.shape[0]) / n_group

                def margin(d: float, bc=bc, br=br,
                           bi=bi) -> torch.Tensor:
                    # d * beta * [(pol(chosen)-ref(chosen)) -
                    # (pol(rejected)-ref(rejected))] with the policy
                    # steered by d*v (BiPO Eq. 3).
                    pol_c = steered_logprobs(bc, d)
                    pol_r = steered_logprobs(br, d)
                    return d * ((pol_c - ref_c[bi]) - (pol_r - ref_r[bi]))

                if direction == "sampled":
                    step_margin = margin(d_sampled)
                    loss = -F.logsigmoid(beta * step_margin).mean()
                    m = step_margin.detach().mean().double() * w
                    step_mean = m if step_mean is None else step_mean + m
                else:
                    fwd_margin = margin(+1.0)
                    rev_margin = margin(-1.0)
                    loss = -(
                        F.logsigmoid(beta * fwd_margin)
                        + F.logsigmoid(beta * rev_margin)
                    ).mean()
                    mf = fwd_margin.detach().mean().double() * w
                    mr = rev_margin.detach().mean().double() * w
                    fwd_mean = mf if fwd_mean is None else fwd_mean + mf
                    rev_mean = mr if rev_mean is None else rev_mean + mr
                (loss if w == 1.0 else loss * w).backward()
                dl = loss.detach().double() * w
                loss_t = dl if loss_t is None else loss_t + dl
            grad_t = grad_clipped_t = None
            if vec.grad is not None:
                if max_grad_norm is not None:
                    # One norm computation serves both the clip and the
                    # telemetry: clip_grad_norm_'s RETURN is the pre-clip
                    # global norm (never recomputed + float()d), and the
                    # optimizer consumes the clipped gradient. The log
                    # keeps both sides of the clip.
                    grad_t = nn.utils.clip_grad_norm_([vec], max_grad_norm)
                    if log_fn:
                        grad_clipped_t = vec.grad.detach().norm()
                else:
                    grad_t = vec.grad.detach().norm()
            finite_t = torch.isfinite(loss_t)
            if grad_t is not None:
                finite_t = finite_t & torch.isfinite(grad_t)
            if not finite_t.item():  # the ONE per-step device sync
                # Abort BEFORE the poisoned step is applied: a diverged fit
                # must fail loudly, never return a vector trained past a
                # NaN/inf (the sweep records the trial as failed). Flush
                # first so the clean steps' telemetry still lands; the
                # failure path may sync freely.
                _flush()
                bad_loss = float(loss_t)
                bad_grad = float(grad_t) if grad_t is not None else None
                raise BidpoNonFiniteError(
                    f"non-finite fit at optimizer step {step + 1} (epoch "
                    f"{epoch}): loss={bad_loss!r}, grad_norm={bad_grad!r} — "
                    "aborting instead of returning a vector")
            if lr_at is not None:
                lr_now = lr_at(step)
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
            opt.step()
            step += 1
            rec: dict = {"bidpo/loss": loss_t}
            if grad_t is not None:
                rec["bidpo/grad_norm"] = grad_t
            if log_fn:
                rec["bidpo/vec_norm"] = vec.detach().norm()
                rec["epoch"] = epoch
                if lr_at is not None:
                    rec["bidpo/lr"] = lr_now
                if grad_clipped_t is not None:
                    rec["bidpo/grad_norm_clipped"] = grad_clipped_t
                if direction != "sampled":
                    if fwd_mean is not None:
                        rec["bidpo/fwd_margin"] = fwd_mean
                    if rev_mean is not None:
                        rec["bidpo/rev_margin"] = rev_mean
            pending.append((step, rec))
            if epoch_margins is not None:
                if direction == "sampled":
                    if step_mean is not None:
                        epoch_margins[d_sampled].append(step_mean)
                elif fwd_mean is not None and rev_mean is not None:
                    epoch_margins[+1.0].append(fwd_mean)
                    epoch_margins[-1.0].append(rev_mean)
            if step % log_every == 0:
                _flush()
            if probe_cb is not None and step % probe_every == 0:
                # The probe cadence is a flush point too: the probe's
                # consumer (FitMonitor.on_probe) emits its own records at
                # this step and syncs anyway, so flushing here keeps the
                # merged stream in step order at no extra sync cost.
                _flush()
                probe_cb(step, _live_probe_object(step))
            if max_steps is not None and step >= max_steps:
                # Hard step budget: stop mid-epoch, skip the stop-rule
                # check (this epoch's margin evidence is partial), record
                # the truth in steps_run/epochs_run.
                budget_exhausted = True
                break
        if budget_exhausted:
            break
        epochs_run = epoch + 1
        if epoch_margins is not None:
            # sum(v)/len(v) over 0-dim fp64 tensors keeps the historical
            # sequential-sum op order (bit-identical means); the two-sided
            # gate reduces to ONE device bool — a single sync per epoch,
            # only when the stop rule is armed.
            means = {d: sum(v) / len(v)
                     for d, v in epoch_margins.items() if v}
            if len(means) == 2 and (
                    torch.stack([means[+1.0], means[-1.0]])
                    >= stop_margin).all().item():
                if log_fn:
                    pending.append(
                        (step, {"bidpo/early_stop_epoch": epoch,
                                "bidpo/early_stop_margin_fwd": means[+1.0],
                                "bidpo/early_stop_margin_rev": means[-1.0]}))
                break

    _flush()  # end of fit: the final materialisation point
    return SteeringVector(
        vectors={layer: vec.detach().float().cpu()},
        method="bidpo",
        model_name=model.config._name_or_path,
        instruction_id=instruction_id,
        metadata={
            "beta": beta,
            "lr": lr,
            "betas": list(betas),
            "eps": eps,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "epochs_run": epochs_run,
            "steps_run": step,
            "accum_steps": accum_steps,
            "effective_batch": batch_size * accum_steps,
            "max_steps": max_steps,
            "max_grad_norm": max_grad_norm,
            "total_steps_planned": total_steps,
            "lr_schedule": lr_schedule,
            "warmup_steps": warmup_resolved,
            "warmup_frac": warmup_frac,
            "stop_pref_prob": stop_pref_prob,
            "n_examples": len(examples),
            "direction": direction,
            "positions": positions,
            "response_header_len": response_header_len,
            "skip_bos": skip_bos,
            "final_loss": history[-1] if history else None,
            "min_loss": min(history) if history else None,
            "final_grad_norm": last_grad_norm,
            "max_grad_norm_seen": max_grad_seen,
            "loss_history_head": history[:5],
        },
    )
