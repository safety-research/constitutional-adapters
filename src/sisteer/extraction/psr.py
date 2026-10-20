"""PSR — Probe-gated steering (Heyman & Vandeputte, arXiv:2605.03907).

The object: per layer l, a FIXED direction d_l (the mean difference of
positive vs negative activations, from the capture/diffmean machinery)
and a LEARNED token-dependent gate g_l(h) = ReLU(w_l . h + b_l) — a
1-layer ReLU probe over the token's residual state. The intervention is

    h  <-  h + scale * g_l(h) * d_l

so the steering STRENGTH is input-dependent (the point of the row) while
the steering DIRECTION is fixed at extraction time.

Trainers (train_psr):
- loss="mse"  (the paper-faithful arm `psr`, verified against the paper
  LaTeX + the Nokia-Bell-Labs official code): match the steered no-SI
  student's residual activations to the SI-prompted teacher's activations
  on completion-token positions, SUMMED over the intervention layer AND
  every subsequent layer (their layers_to_imitate="all"; scoring only the
  intervention layer was a repo bug), plus the anti-dead-gate
  regularizer reg_coeff * sum_l mean_batch[max(0, 1 - sum_i gate_l(h_i))]
  with reg_coeff 0.1 (their code-only value). BOTH the gate AND the
  direction train (their code: the direction is a trainable Linear; the
  earlier mean-diff FREEZE was a hybrid the paper never ran). The
  mean-diff is kept as the direction's INIT only — an init-only
  deviation: the paper text is silent and their code uses default init;
  starting from the diffmean vector is strictly more informed.
- loss="kl"   (`psr_kl`, `psr_kl_all`): gate AND direction trained
  jointly under train_object_kl's forward-KL objective (single layer or
  all layers per the layer axis) — DELIBERATE non-faithful variants,
  labeled so. Since 2026-08-20 the KL path carries the SAME
  anti-dead-gate regularizer as the mse path (reg_coeff 0.1, threaded
  through train_object_kl), and the all-layer variants of BOTH
  objectives zero-init — see "Confound removed" below. The SINGLE-layer
  KL variants still INITIALIZE the direction from the mean-diff (one
  layer's vector is the standard informed start). The ALL-layer
  variants (psr_mse_all, psr_kl_all, psr_kl_all_hard) ZERO-init
  instead, like every other all-layer arm
  (multilayer.py): stacking ~32 simultaneous full-strength diffmean
  vectors (per-layer norms up to ~17) under the wide-open init gate put
  a fit's step-0 KL at 9.28 against a measured 0.148 no-op floor, and
  the FIRST AdamW step (lr 5e-3) saturated the deep-layer gates so that
  w.h + b < 0 at every token — ReLU dead, zero gradient to the gate AND
  to that layer's direction, forever (the KL path then carried no
  anti-dead-gate regularizer). Measured on the saved artifacts (numbers read off the
  blobs themselves, correcting the 9d84801b commit message): a contiguous
  deep block of gates frozen at gate_b=0.924214 (identical to 6 decimals
  per object — AdamW's exact impulse response to one nonzero gradient
  followed by zero gradients, magnitude-invariant because
  m-hat/sqrt(v-hat) cancels; the impulse fit holds for gate_b, NOT for
  gate_w, whose frozen |w| is ~0.0286) on BOTH SIs — 18/32 layers
  (13..30) on safety_refusal, 11/32 (19, 20, 22..30) on si_constitution —
  final losses 4.82/5.72 ~ 30x WORSE than the no-op floor. Both
  psr_mse_all objects were pristine (0 frozen gates, 32/32 distinct
  gate_b): exactly what "the mse path carries reg_coeff 0.1 gate_reg and
  the KL path carries nothing" predicts. With
  zero-init, d=0 makes step 0 an exact no-op, the gate stays alive at
  g==1, and directions grow from zero under the nonzero gradient
  g * dL/d(add). check_psr_gates below is the fail-loudly guard that
  keeps a dead-gate object from ever being treated as trained again.

Confound removed (2026-08-20): the two protective treatments
used to be SPLIT across the objectives — zero-init only on the all-layer
KL arms, the gate regularizer only on the mse path — so an MSE-vs-KL
difference conflated the objective with its protection. Now BOTH
objectives get BOTH treatments: the all-layer variants of mse AND kl
zero-init their directions (single-layer arms keep the informed
mean-diff start on both), and the anti-dead-gate regularizer applies on
both paths at the same reg_coeff 0.1 default (configurable per arm).
An MSE-vs-KL gap therefore isolates the OBJECTIVE alone. The
regularizer makes gate death less likely; it does not resurrect a gate
that is already fully closed (ReLU's zero gradient silences the
regularizer too), so check_psr_gates stays load-bearing on every path.

23k-pair wave failure (2026-08-22) and the two fixes it forced: at the
full pool scale (~23,002 matched pairs = 5,165 steps/epoch, 6 epochs,
fixed lr 5e-3), BOTH all-layer arms failed their end-of-training guards
on every SI, despite zero-init and the shared regularizer. The
trajectories (the saved __metrics.jsonl) pin the mechanism:

- The FIRST AdamW step blows the loss up 60-85x (mse 0.019 -> 1.62,
  KL 0.008 -> 0.51): at step 1 AdamW's m-hat/sqrt(v-hat) is +-1, so
  every coordinate of every parameter moves by exactly lr — 32
  zero-init directions become 32 simultaneous junk vectors of norm
  lr*sqrt(d) under wide-open gates. The zero-init fixed step 0; the
  step-1 impulse was untouched.
- In the exploded regime the objective's cheapest move is to close
  gates (less junk injected), and one sign-consistent AdamW step can
  swing w.h by ~lr * ||h||_1 (thousands of coordinates moving lr each,
  signs correlated with h) — tens of units against the b=1 margin. Gate
  deaths accumulate as a RATCHET: reg (which pays exactly 1 per fully
  dead layer) reaches ~10 dead layers by step ~350-900 of epoch 0 and
  climbs monotonically to 19-27 by the end (26/32, 17/32, 27/32 dead at
  the psr_mse_all guard; the KL arms plateaued at 34-38x the no-op
  floor with ~20 layers dead). More steps = more deaths, which is why
  stage-2-scale fits (hundreds of steps) passed and 31k-step fits died.
- The mse loop also spikes (single-step losses up to 1.9e8-1.7e11,
  steering feedback compounding over 30 scored layers); one such
  gradient poisons AdamW's second moment for ~1k steps.
- Nothing ever recovers: fixed-lr AdamW never anneals (the KL held-out
  loss goes 0.015 -> 7.3 within epoch 0 and stays 7.1-13 for five more
  epochs), and a fully closed gate is unreachable by BOTH gradient
  paths — objective and regularizer alike go through the dead ReLU.

Fixes, both in this file (train_object_kl/lockstep live elsewhere):
(1) the anti-dead-gate regularizer gained a straight-through LEAK
(GATE_REG_LEAK): its VALUE is unchanged everywhere (bit-identical
losses and metrics), but where a row's gate mass sits below 1 its
gradient also flows to the ReLU's NEGATIVE pre-activations, so a fully
closed gate keeps receiving a reopening gradient — death is no longer
permanent. It rides obj.gate_reg_term into every trainer (mse loop,
train_object_kl, train_lockstep) without touching them, and it is
exactly zero-effect while gates are healthy. (2) train_psr's OWN mse
loop (the psr/psr_mse_all trainer) got linear LR warmup + cosine decay
and gradient clipping by default: warmup removes the step-1 impulse
(AdamW's v is calibrated before full-size steps happen), decay anneals
the end of training instead of plateauing at the fixed-lr noise floor,
clipping caps the spike steps. The KL arms' optimizer loop lives in
train_object_kl and is NOT scheduled here — train_psr refuses schedule
arguments on the KL path rather than silently dropping them.

Defaults chosen here (the gate is the ReLU probe named in
docs/method_specs_s2.md):
- gate init w=0, b=1: the gate starts as the CONSTANT 1, i.e. PSR begins
  as the plain mean-diff steering vector at scale 1, and — unlike a
  zero-init ReLU, whose gradient is dead at exactly 0 — gradients flow
  from step 1.
- the gate applies at every position (no BOS skip): the gate is learned,
  so down-weighting BOS/outlier tokens is its job, not a mask's.
"""

from __future__ import annotations

import math
import random
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn

from ..hooks import POSITION_MODES, PositionMaskedFit, SteeringHandle
from ..models import get_decoder_layers, hidden_size, num_layers
from .diffmean import extract_caa
from .lora_modules import INIT_SCHEMES, init_generator, lecun_sigma
from .fitckpt import (
    cuda_rng_states,
    load_checkpoint,
    resolve_checkpoint_kwargs,
    restore_cuda_rng,
    save_checkpoint,
    to_cpu,
)
from .distill import (DistillExample, EarlyStopper, EarlyStopSpec,
                      TokenizeProgress, _encode_aligned,
                      steer_prompt_boundaries)
from .prefix_cache import completion_token_mask_suffix, prefix_cache_for_batches
from .traindata import Batch, collate


class PSRObject(PositionMaskedFit):
    """Mean-diff direction + ReLU-probe gate, one (d, w, b) per layer.

    Position-masked fits (PositionMaskedFit): the intervention is a
    per-position residual-stream addition h + g(h)·d at the decoder-layer
    output — the same tensor hooks.steering edits — so restricting it to
    the handle's mask is EXACT; masked positions keep bit-identical hidden
    states. apply() yields the live mask handle (None on the default
    all-positions fit) so trainers tie the loss gather to the same mask.
    The gate stash (_last_gate) keeps EVERY position's pre-steering gate —
    the mse regularizer masks its pooling separately (train_psr)."""

    kind = "psr"

    def __init__(self, model, layers: list[int],
                 train_direction: bool = False,
                 positions: str = "all",
                 skip_bos: bool = False,
                 response_header_len: int | None = None,
                 *, init_scheme: str = "zero", seed: int = 0):
        d = hidden_size(model)
        self.layers = list(layers)
        self.train_direction = train_direction
        # FD 7 (2026-09-29): the paper's object is ZERO-init (directions
        # v = 0, gate w = 0, gate bias b = 1 — the untrained object is the
        # unsteered model with an open gate) and that stays the default,
        # byte-identical. A run override may select
        # "random_lecun_1e-3": v ~ N(0, σ²), w ~ N(0, σ²), σ = 1e-3/√d, b = 1,
        # drawn per layer from a stream keyed by (seed, layer) so the same
        # seed gives the same init in the search trial, the production fit
        # and the constitution/control twins of a seed.
        if init_scheme not in INIT_SCHEMES:
            raise ValueError(
                f"init_scheme must be one of {INIT_SCHEMES}, got {init_scheme!r}")
        self.init_scheme = str(init_scheme)
        self.seed = int(seed)
        # Position masking (parity with the vector arm's SteeringHandle). The
        # ORIGINAL PSR hook masked nothing and never skipped BOS — that stays
        # the default (positions="all", skip_bos=False) so training and every
        # existing inference run are byte-identical. positions="response" here
        # is the SERVE-time deployment mask. Whether it is on-distribution
        # depends on the FIT leg, which is a SEPARATE axis (PositionMaskedFit,
        # 2026-08-20): a Class A object (fit_positions="all") trained with the
        # intervention at every position and only the LOSS restricted to
        # completion tokens, so serving it response-only is off-distribution
        # by construction; a Class C object (set_fit_positions("response", H))
        # trained with the intervention itself masked to the header+response
        # positions — prompt positions bit-identical to the unsteered model
        # via _fit_masked — so response serving IS its training distribution.
        # (This comment once described the pre-2026-08-20 world, where every
        # fit was all-position; that claim misread as current cost a real
        # audit cycle — keep the two axes distinct here.)
        #
        # Pre-existing ASYMMETRY vs. the vector arm: the vector arm defaults to
        # skip_bos=True (HyperSteer excludes BOS); PSR's gate is LEARNED, so
        # down-weighting BOS is the gate's job, not a mask's — hence skip_bos
        # defaults False here. It is settable for parity, but changing it
        # departs from how the object was trained.
        if positions not in POSITION_MODES:
            raise ValueError(f"positions must be one of {POSITION_MODES}")
        self.positions = positions
        self.skip_bos = skip_bos
        # See SteeringHandle: when set, prompt_len is DERIVED per prefill as
        # prefill_width - H (constant-length assistant-header suffix under forced
        # LEFT padding), the generation-time response boundary.
        self.response_header_len = response_header_len
        self.prompt_len = 0
        self._seen = 0            # tokens consumed so far in an incremental generate
        # Authoritative global offset from the KV cache, mirroring
        # SteeringHandle. Without it, an apply() entered MID-GENERATION restarts
        # positions at 0, so the first decode step looks like global position 0
        # -- which skip_bos then masks, silently dropping steering on the first
        # generated token — the same bug class fixed in SteeringHandle.
        self._cache_len: int | None = None
        self._cur_offset = 0
        self._attn_mask: torch.Tensor | None = None  # [B, S] 2D padding mask
        self.directions: dict[int, nn.Parameter] = {}
        self.gate_w: dict[int, nn.Parameter] = {}
        self.gate_b: dict[int, nn.Parameter] = {}
        sigma = lecun_sigma(d) if self.init_scheme != "zero" else None
        for l in self.layers:
            if sigma is None:
                v, w = torch.zeros(d), torch.zeros(d)
            else:
                g = init_generator("psr", self.seed, int(l))
                v = torch.randn(d, generator=g) * sigma
                w = torch.randn(d, generator=g) * sigma
            self.directions[l] = nn.Parameter(v, requires_grad=train_direction)
            self.gate_w[l] = nn.Parameter(w)
            self.gate_b[l] = nn.Parameter(torch.ones(1))
        self._decoder = get_decoder_layers(model)
        # Per-layer gate outputs of the LAST forward under apply(), stashed
        # by the hook on the PRE-steering hidden states (the regularizer
        # must see the gates the forward actually used, never a
        # recomputation from post-steering captures). Cleared when the
        # apply() context exits.
        self._last_gate: dict[int, torch.Tensor] = {}
        # The matching PRE-ReLU activations (w.h + b) of the same forward,
        # raw (never position-masked). The regularizer's straight-through
        # leak reads these: post-ReLU gates carry no gradient at a closed
        # gate, which is exactly how the 23k-pair wave's dead gates became
        # permanent (module docstring). Cleared with _last_gate.
        self._last_pre: dict[int, torch.Tensor] = {}

    def set_directions(self, directions: dict[int, torch.Tensor]) -> None:
        for l in self.layers:
            self.directions[l].data = directions[l].detach().float().clone()

    @property
    def trainable(self) -> list[nn.Parameter]:
        params = [self.gate_w[l] for l in self.layers]
        params += [self.gate_b[l] for l in self.layers]
        if self.train_direction:
            params += [self.directions[l] for l in self.layers]
        return params

    def n_params(self) -> int:
        return sum(p.numel() for p in self.trainable)

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def describe(self) -> dict:
        """The registry-facing description (the ``params_json`` names;
        the ReLU steer has no LoRA axes and reports its layer list under
        ``layers_by_module["residual"]``) — the same dict shape
        :meth:`MultiLayerLoRAObject.describe` returns."""
        return {
            "object_kind": "relu_steer",
            "lora_rank": None, "lora_modules": None, "lora_set_id": None,
            "per_expert_rank": None,
            "init_scheme": self.init_scheme,
            "trainable_params": int(self.n_params()),
            "layers_by_module": {"residual": [int(l) for l in self.layers]},
        }

    def gate_pre(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """The gate's PRE-ReLU activation w_l . h + b_l, one scalar per
        token: [B, S]. Split out of :meth:`gate` so the hook can stash it
        for the regularizer's straight-through leak (a closed gate's only
        remaining gradient path)."""
        w = self.gate_w[layer].to(device=hidden.device)
        b = self.gate_b[layer].to(device=hidden.device)
        return hidden.float() @ w + b

    def gate(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """g_l(h) = ReLU(w_l . h + b_l), one scalar per token: [B, S]."""
        return F.relu(self.gate_pre(layer, hidden))

    def gate_reg_term(self, attention_mask: torch.Tensor,
                      mask_handle: SteeringHandle | None = None,
                      ) -> torch.Tensor:
        """The anti-dead-gate regularizer term for the CURRENT forward
        (coefficient applied by the caller). Must be called INSIDE
        apply() — the pre-steering gate stash (``_last_gate``) is cleared
        when the context exits — and AFTER the forward that filled it.

        Pools gate mass over REAL ∩ STEERED positions: ``attention_mask``
        is the student batch's [B, S] non-pad mask, and a live
        ``mask_handle`` (a masked fit's boundary handle, prompt lens
        already set) restricts it to the positions the edit actually
        applies at — a gate alive only where the edit never applies
        steers nothing.

        This is the ONE implementation both training paths call: the mse
        loop and the KL loop (train_object_kl, and train_lockstep for the
        lockstep-compatible KL arms). Sharing it is what keeps the
        solo-vs-lockstep bitwise contract intact for the regularized
        arms — two transcriptions of the same pooling could diverge in
        op order and break exact equality."""
        if not self._last_gate:
            raise RuntimeError(
                "gate_reg_term called with an empty gate stash — it must "
                "run inside apply(), after the forward that populates "
                "_last_gate (the stash is cleared at context exit).")
        gates = {l: self._last_gate[l] for l in self.layers}
        pre = {l: self._last_pre[l] for l in self.layers}
        valid = attention_mask.bool()
        if mask_handle is not None:
            b, s = attention_mask.shape
            valid = valid & mask_handle._position_mask(
                b, s, attention_mask.device).bool()
        return gate_regularizer(gates, valid, pre_acts=pre)

    # -- position bookkeeping (mirrors hooks.SteeringHandle) -------------------

    def _needs_tracking(self) -> bool:
        return (self.positions != "all" or self.skip_bos
                or self.response_header_len is not None)

    def _track_model_inputs(self, module, args, kwargs):
        mask = kwargs.get("attention_mask")
        self._attn_mask = mask if (mask is not None and mask.dim() == 2) else None
        # Also capture the KV-cache length here -- the ONLY place
        # past_key_values is visible (per-layer it is a 4D causal mask or None).
        # Authoritative global offset; without it an apply() entered
        # MID-GENERATION restarts at 0 and skip_bos then masks the first
        # generated token, silently dropping steering on it.
        pkv = kwargs.get("past_key_values")
        _get = getattr(pkv, "get_seq_length", None)
        self._cache_len = int(_get()) if _get is not None else None
        return None

    def _track_positions(self, module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        seq_len = hidden.shape[1]
        if seq_len > 1:  # PREFILL / full-sequence forward: new sequence
            self._seen = 0
            if self.response_header_len is not None:
                # Boundary from THIS prefill's padded width; the last H columns
                # are the assistant header under forced LEFT padding.
                self.prompt_len = max(seq_len - self.response_header_len, 0)
        if self._cache_len is not None:
            # authoritative: survives entering the context mid-generation
            self._cur_offset = self._cache_len
            self._seen = self._cache_len + seq_len
        else:
            self._cur_offset = self._seen
            self._seen += seq_len
        self._cache_len = None
        return None

    def _bos_positions(self, batch: int, device) -> torch.Tensor:
        """[B] global index of the first REAL (non-pad) position per row —
        argmax of the 2D attention mask (true BOS under LEFT/right padding)."""
        am = self._attn_mask
        if am is not None and am.shape[0] == batch:
            return am.argmax(dim=1).to(device)
        return torch.zeros(batch, dtype=torch.long, device=device)

    def _position_mask(self, batch: int, seq_len: int, device) -> torch.Tensor | None:
        """[B, S] float mask of positions to steer, or None for all. GLOBAL
        coordinates only (generation path); PSR training uses positions="all"
        and a separate label mask, so no per-sample local branch is needed."""
        if not self._needs_tracking():
            return None
        idx = torch.arange(
            self._cur_offset, self._cur_offset + seq_len, device=device
        )  # [S] global positions
        if self.positions == "response":
            mask = idx >= self.prompt_len
        elif self.positions == "prompt":
            mask = idx < self.prompt_len
        else:  # "all" (here only when skip_bos) / "front" fallback
            mask = torch.ones_like(idx, dtype=torch.bool)
        mask = mask[None, :].expand(batch, seq_len)
        if self.skip_bos:
            mask = mask & (idx[None, :] != self._bos_positions(batch, device)[:, None])
        return mask.float()

    def _make_hook(self, layer: int, scale: float,
                   mask_handle: SteeringHandle | None = None):
        def hook(_mod, _args, output):
            if scale == 0.0:  # exact no-op, bit-identical to unsteered
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            z = self.gate_pre(layer, hidden)  # [B, S], PRE-steering hidden
            g = F.relu(z)
            mask = self._position_mask(hidden.shape[0], hidden.shape[1],
                                       hidden.device)
            if mask is not None:
                # Zero the gate at masked positions so they contribute EXACTLY
                # nothing to the steering addition (and to the regularizer's
                # gate stash). None -> unchanged, so positions="all" is
                # byte-identical to the original hook.
                g = g * mask.to(g.dtype)
            self._last_gate[layer] = g  # grads flow to gate_w/gate_b
            self._last_pre[layer] = z   # raw pre-ReLU, for the reg's leak
            d = self.directions[layer].to(device=hidden.device)
            add = (scale * g).unsqueeze(-1) * d  # [B, S, d]
            steered = hidden + add.to(hidden.dtype)
            steered = self._fit_masked(mask_handle, steered, hidden)
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        return hook

    @contextmanager
    def apply(self, model, scale: float = 1.0,
              positions: str | None = None, prompt_len: int | None = None):
        """Steering context. Two ways to restrict positions, and they meet in
        the SAME mask home (``_position_mask``):

        - constructed state (``PSRObject(..., positions=, skip_bos=,
          response_header_len=)``) — the serve-cell configuration;
        - a PER-CALL override (longctx generation-only ablation):
          ``apply(positions="response", prompt_len=width)`` steers only
          global positions >= prompt_len for THIS context, restoring the
          constructed state on exit. Only "all"/"response" are defined for
          the per-call form (fail loudly). Training paths pass
          neither and keep the constructed default untouched.

        Position bookkeeping mirrors hooks.SteeringHandle exactly — the
        event-based tracker (prefill resets; the model-boundary KV length
        is the authoritative offset), never a seq_len==1 "is decoding"
        guess."""
        if positions is not None and positions not in ("all", "response"):
            raise ValueError(
                f"PSRObject.apply positions must be 'all' or 'response', "
                f"got {positions!r}"
            )
        saved = (self.positions, self.prompt_len, self._seen,
                 self._cur_offset, self._attn_mask, self._cache_len)
        if positions is not None:
            self.positions = positions
        if prompt_len is not None:
            self.prompt_len = prompt_len
        mask_handle = self._fit_mask_handle(model)
        handles = []
        if self._needs_tracking():
            # The tracker must fire once per forward BEFORE any steered layer.
            # Layer 0 always runs; the 2D padding mask only exists at the model
            # boundary (per-layer masks are 4D/None), so grab it there.
            handles.append(self._decoder[0].register_forward_pre_hook(
                self._track_positions, with_kwargs=True))
            handles.append(model.register_forward_pre_hook(
                self._track_model_inputs, with_kwargs=True))
        handles += [
            self._decoder[l].register_forward_hook(
                self._make_hook(l, scale, mask_handle))
            for l in self.layers
        ]
        try:
            yield mask_handle
        finally:
            for h in handles:
                h.remove()
            if mask_handle is not None:
                mask_handle.remove()
            self._last_gate = {}
            (self.positions, self.prompt_len, self._seen,
             self._cur_offset, self._attn_mask, self._cache_len) = saved
            self._last_pre = {}

    def to_(self, device) -> None:
        for group in (self.directions, self.gate_w, self.gate_b):
            for p in group.values():
                p.data = p.data.to(device)

    def save(self, path: Path) -> None:
        """Persist the object AND a provenance sidecar.

        The sidecar exists because its absence cost three rounds of cluster
        archaeology to answer "which model trained this". Inferring from the
        config is NOT sound: the build can run on a cluster whose config names
        a different mirror than the one you are reading. Two mirrors of this
        base model have IDENTICAL WEIGHTS but DIFFERENT chat templates, so the
        training template is not recoverable from the tensors -- it has to be
        recorded at fit time or it is gone.

        Written as ``<path>.provenance.json`` so it never collides with the
        torch blob and an older loader ignores it.
        """
        torch.save(
            {"kind": self.kind,
             "layers": self.layers,
             "train_direction": self.train_direction,
             # The FIT's mask leg is part of the artifact's identity (the
             # DiffMeanObject/LoReFT precedent, 2026-08-21): before this key
             # existed a response-fit PSR re-hydrated ALL-POSITION and would
             # silently run the all-token method under a decode-only label.
             # NOTE this is the fit leg (PositionMaskedFit), not the
             # serve-time deployment attr self.positions, which stays a
             # runtime option and is never persisted.
             "positions": self.fit_positions,
             "response_header_len": self.fit_response_header_len,
             "directions": {str(l): self.directions[l].detach().float().cpu()
                            for l in self.layers},
             "gate_w": {str(l): self.gate_w[l].detach().float().cpu()
                        for l in self.layers},
             "gate_b": {str(l): self.gate_b[l].detach().float().cpu()
                        for l in self.layers},
             # Only the non-default init scheme rides (FD 7 run override);
             # the paper's zero-init objects keep their exact key set.
             **({"init_scheme": self.init_scheme}
                if self.init_scheme != "zero" else {})},
            path,
        )
        try:
            import json as _json
            from ..models import chat_template_hash
            meta = {
                "kind": self.kind,
                "layers": list(self.layers),
                "model_name": getattr(self, "_prov_model_name", None),
                "model_revision": getattr(self, "_prov_model_revision", None),
                "chat_template_hash": getattr(self, "_prov_template_hash", None),
            }
            Path(str(path) + ".provenance.json").write_text(
                _json.dumps(meta, indent=2))
        except Exception:
            # Provenance is diagnostic, never load-bearing: a failure here must
            # not lose a trained object.
            pass

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "PSRObject":
        layers = [int(l) for l in blob["layers"]]
        obj = cls(model, layers, train_direction=blob.get("train_direction", False),
                  init_scheme=str(blob.get("init_scheme", "zero")))
        for l in layers:
            obj.directions[l].data = blob["directions"][str(l)].clone()
            obj.gate_w[l].data = blob["gate_w"][str(l)].clone()
            obj.gate_b[l].data = blob["gate_b"][str(l)].clone()
        # Backward-compatible fit-leg restore (the DiffMeanObject pattern):
        # an artifact with no positions key predates the persist and loads
        # as its historical all-token fit, byte-identical behavior.
        if blob.get("positions", "all") != "all":
            obj.set_fit_positions(blob["positions"],
                                  blob.get("response_header_len"))
        return obj


def init_psr_directions(model, tok, pairs, layers: list[int]) -> dict[int, torch.Tensor]:
    """Mean-diff directions from contrastive pairs (CAA pooling over
    completion tokens, via the existing capture machinery)."""
    from ..data.pairs import pairs_to_caa_inputs

    prompts, pos, neg = pairs_to_caa_inputs(tok, pairs)
    sv = extract_caa(model, tok, prompts, pos, neg, layers,
                     position="completion_mean")
    return {l: sv.vectors[l] for l in layers}


@contextmanager
def _capture_hidden(model, layers: list[int]):
    """Like capture.capture_layer_outputs but WITHOUT detaching, so the
    student-side capture keeps gradients flowing to the gate. Registered
    inside obj.apply's context, so it sees the post-PSR (steered) hidden
    states (later-registered forward hooks receive earlier hooks'
    modified output)."""
    captured: dict[int, torch.Tensor] = {}
    decoder = get_decoder_layers(model)
    handles = []

    def make(idx):
        def hook(_mod, _args, output):
            captured[idx] = output[0] if isinstance(output, tuple) else output

        return hook

    for idx in layers:
        handles.append(decoder[idx].register_forward_hook(make(idx)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def psr_loss_layers(obj: PSRObject, model) -> list[int]:
    """S-PSR's MSE loss layer set: the (earliest) intervention layer AND
    every subsequent layer to the top of the stack (their
    layers_to_imitate="all")."""
    return list(range(min(obj.layers), num_layers(model)))


# The regularizer's straight-through leak slope: how much reopening
# gradient a NEGATIVE gate pre-activation receives (per valid token) while
# its row's gate mass sits below 1. The VALUE of the term never changes —
# the leak enters as leak * (neg - neg.detach()), identically zero in the
# forward — so metrics, guards and every already-passing loss stay
# bit-identical; only the backward differs, and only in the collapse
# regime. Why it exists: in the 23k-pair wave (2026-08-22, module
# docstring) gates that closed in epoch 0 stayed closed for 30k steps
# because a closed ReLU zeroes BOTH gradient paths (objective and
# regularizer); with the leak, a fully dead layer's (w, b) keep receiving
# a reopening gradient — and since a dead layer's objective gradient is
# exactly zero, the leak is that layer's ONLY gradient, so AdamW's
# sign-normalized steps march it back to life at ~lr per step no matter
# how small this slope is. 0.01 is the conventional leaky-ReLU slope;
# nothing downstream is calibrated against its magnitude.
GATE_REG_LEAK = 0.01


def gate_regularizer(gates: dict[int, torch.Tensor],
                     valid: torch.Tensor,
                     pre_acts: dict[int, torch.Tensor] | None = None,
                     leak: float | None = None) -> torch.Tensor:
    """Anti-dead-gate term, shared by BOTH objectives since 2026-08-20
    (mse always had it — the paper's code-only 0.1 term; the KL path
    gained it so MSE-vs-KL isolates the objective). Callers assemble the
    inputs through PSRObject.gate_reg_term (the one shared assembly);
    the term itself (coefficient applied by the caller):
    sum over steered layers of mean_batch[ max(0, 1 - sum_{i in valid}
    gate_l(h_i)) ], with the gate values taken from the student forward's
    PRE-steering stash (PSRObject._last_gate) and `valid` the [B, S]
    non-pad position mask. Zero as soon as each row's total gate mass
    reaches 1; a fully dead gate pays exactly 1 per layer.

    ``pre_acts`` (the matching pre-ReLU stash, PSRObject._last_pre) arms
    the straight-through leak: the pooled mass gains
    ``leak * (neg - neg.detach())`` where ``neg`` is the sum of the
    NEGATIVE pre-activations over valid positions — zero in the forward
    (every value this function returns is bit-identical with or without
    it), but in the backward it hands each closed position a reopening
    gradient of ``leak`` whenever the row's mass is below 1. Without it a
    fully closed gate is unrecoverable (the 23k-pair wave's ratchet: the
    ReLU zeroes the objective's gradient AND this term's). ``pre_acts=
    None`` keeps the historical gradient exactly (direct callers that
    only have post-ReLU gates); ``leak=None`` means
    :data:`GATE_REG_LEAK` — resolved at call time so tests can pin the
    module attribute to 0 and reproduce the pre-leak dynamics."""
    if leak is None:
        leak = GATE_REG_LEAK
    v = valid.to(next(iter(gates.values())).dtype)
    per_layer = []
    for l, g in gates.items():
        mass = (g * v).sum(dim=1)
        if pre_acts is not None and leak:
            neg = (pre_acts[l].clamp(max=0.0) * v).sum(dim=1)
            mass = mass + leak * (neg - neg.detach())
        per_layer.append(F.relu(1.0 - mass).mean())
    return torch.stack(per_layer).sum()


class DeadGateError(RuntimeError):
    """A trained PSR's ReLU gates are closed at every probe token on too
    many layers — those layers steer nothing, and at g == 0 the ReLU's
    zero gradient silences the objective AND the anti-dead-gate
    regularizer's post-ReLU path (both objectives carry the term since
    2026-08-20). Since the 23k-pair wave fix the regularizer's
    straight-through leak (GATE_REG_LEAK) gives a closed gate a
    reopening gradient during TRAINING, so death is no longer permanent
    mid-fit — but an object that still ENDS with this many closed gates
    is exactly as broken to serve, so the guard's authority is
    unchanged."""


class ProbePairsPathError(FileNotFoundError):
    """The probe-pairs jsonl the dead-gate guard needs is missing or
    unreadable at the given path — on the cluster, most often because the
    path is grading-host-local or under ``/home`` (node-local disk), which a
    compute node cannot see."""


# The minimum number of probe examples a save/no-save gate verdict may rest
# on. Why 64: a 4-example probe once produced a clean, monotone liveness
# effect — corroborated across two independently-trained objects — that
# vanished and partially reversed at n=32 and had to be retracted. n=32 is
# therefore demonstrably within noise-flip range of n=4; 64 doubles the n
# that overturned it, and at the typical >=50 valid positions per example
# puts every per-layer verdict on thousands of probed positions.
GUARD_PROBE_FLOOR = 64

# Probe forwards run in chunks of this many examples so the guard's memory
# stays flat however many examples are probed (64 examples x 768 positions
# of logits in ONE batch is tens of GB at 9B scale; 8 at a time is not).
_PROBE_CHUNK = 8


def validate_probe_pairs_path(pairs_path: str | Path) -> Path:
    """Fail-fast validation of a probe-pairs jsonl path — call this at the
    TOP of a job script, BEFORE the model loads, so a bad path costs
    milliseconds instead of a GPU allocation.

    Checks the file exists, is a file, is readable, and is non-empty on
    THIS host; raises :class:`ProbePairsPathError` (a ``FileNotFoundError``
    subclass) naming the path and the fix otherwise. Returns the ``Path``
    on success so it can be used inline::

        pairs_path = validate_probe_pairs_path(
            "$SHARED/two_si/pairs/safety_refusal.jsonl")

    On the cluster the pairs must live on SHARED storage
    (``$SHARED/...``): cluster ``/home`` is node-local disk, so a file
    that exists on the grading host or on one node's ``/home`` does not
    exist on the compute node the job lands on."""
    path = Path(pairs_path).expanduser()
    problem = None
    if not path.exists():
        problem = "does not exist on this host"
    elif path.is_dir():
        problem = "is a directory, not a pairs jsonl"
    else:
        try:
            with path.open("rb") as f:
                first = f.read(1)
        except OSError as e:
            problem = f"is not readable ({e.strerror or e})"
        else:
            if not first:
                problem = "is empty (zero bytes) — not a pairs jsonl"
    if problem is None:
        return path
    raise ProbePairsPathError(
        f"probe-pairs file {str(path)!r} {problem}. The dead-gate guard "
        "probes the gates on real teacher/student pairs, so the pairs "
        "jsonl must be readable on the machine the guard runs on. On the "
        "cluster that means SHARED storage — under the cluster's shared "
        "root, $SHARED/... (the site config's CLUSTER_SHARED_ROOT) — never a "
        "grading host path and never cluster /home, which is node-local disk: a file "
        "that exists on the grading host or on one node's /home does not exist "
        "on the compute node the job lands on. Fix: copy the pairs jsonl "
        "to shared storage, pass that path, and call "
        "validate_probe_pairs_path(path) at the top of the job script, "
        "before the model loads, so a bad path fails in milliseconds "
        "instead of after a model load.")


def _as_probe_examples(examples) -> list[DistillExample]:
    """Up-front validation of the guard's ``examples`` argument — runs
    before any model work, so the two historic misuses fail instantly and
    instructively instead of deep in a forward after an expensive load:
    handing the pairs FILE instead of the rendered examples, and handing a
    path that no compute node can see."""
    if isinstance(examples, (str, Path)):
        # Validate the path FIRST: a nonexistent/node-invisible file gets
        # the shared-storage explanation (the commoner, costlier mistake);
        # a readable file gets told how to turn itself into examples.
        path = validate_probe_pairs_path(examples)
        raise TypeError(
            f"examples got the pairs FILE {str(path)!r} (it exists and is "
            "readable here) — but the guard takes the RENDERED probe "
            "pairs, a list[DistillExample], not the file: load the jsonl "
            "rows into sisteer.data.pairs.Pair and render them with "
            "pairs_to_distill_examples(tok, pairs, instruction), then pass "
            "that list.")
    try:
        n = len(examples)
    except TypeError:
        raise TypeError(
            "examples must be a list[DistillExample] (rendered probe "
            f"pairs), got {type(examples).__name__}") from None
    if n == 0:
        raise ValueError(
            "examples is empty — the gate probe needs real teacher/student "
            "pairs (render them with pairs_to_distill_examples)")
    return list(examples)


def psr_gate_liveness(model, tok, *, obj: PSRObject,
                      examples: list[DistillExample],
                      max_len: int = 768,
                      n_probe: int = GUARD_PROBE_FLOOR) -> dict[int, float]:
    """Per-layer gate liveness: the fraction of valid (non-pad) token
    positions at which the gate is OPEN (ReLU(w_l . h + b_l) > 0), measured
    on no-grad student forwards with the object APPLIED — so the gate
    values are exactly the ones the intervention uses (the pre-steering
    stash ``_last_gate``, with earlier layers' steering already feeding the
    later layers' hidden states, as in training and eval).

    Call contract — ``obj`` and ``examples`` are KEYWORD-ONLY: this module
    follows sisteer's ``(model, tok, ...)`` convention while
    ``load_object`` takes its PATH first, and a ported call once died twice
    on guessed argument order. ``examples`` are the RENDERED teacher/student
    pairs (``pairs_to_distill_examples``), not the pairs file; on a cluster
    the pairs jsonl must live on shared storage (see
    :func:`validate_probe_pairs_path`, and call it before the model loads).

    Exactly ``examples[:n_probe]`` are probed, in memory-flat chunks. If
    FEWER than ``n_probe`` examples are supplied this raises instead of
    silently shrinking the probe: a probe that quietly ran on n=4 once
    produced a clean monotone effect, corroborated across two
    independently-trained objects, that vanished and partially reversed at
    n=32 (see ``GUARD_PROBE_FLOOR``). To deliberately probe fewer, pass
    ``n_probe=len(examples)`` explicitly — the thinness is then written at
    the call site instead of happening silently.

    Liveness 0.0 identifies a DEAD layer exactly, not heuristically, OVER
    THE PROBED POSITIONS: the hook's additive term ``scale * g * d`` is
    identically zero there, and the ReLU's zero gradient at g == 0 means
    gate_w, gate_b AND that layer's direction receive no gradient through
    the steering path — a layer open at even one token still gets gradient
    and can move."""
    examples = _as_probe_examples(examples)
    if len(examples) < n_probe:
        raise ValueError(
            f"psr_gate_liveness: n_probe={n_probe} but only "
            f"{len(examples)} example(s) supplied — refusing to silently "
            "shrink the probe (a probe that quietly shrank to n=4 once "
            "produced a monotone result that vanished at n=32). Supply "
            f"more pairs, or pass n_probe={len(examples)} explicitly to "
            "accept a deliberately thinner probe.")
    device = next(model.parameters()).device
    obj.to_(device)
    probe = examples[:n_probe]
    open_ct = {l: 0 for l in obj.layers}
    valid_ct = 0
    for i in range(0, len(probe), _PROBE_CHUNK):
        chunk = probe[i:i + _PROBE_CHUNK]
        encoded = [_encode_aligned(tok, ex, max_len) for ex in chunk]
        batch = collate([s for _, s in encoded], tok.pad_token_id).to(device)
        with torch.no_grad(), obj.apply(model, scale=1.0):
            model(input_ids=batch.input_ids,
                  attention_mask=batch.attention_mask)
            gates = {l: obj._last_gate[l] for l in obj.layers}
        valid = batch.attention_mask.bool()
        valid_ct += int(valid.sum())
        for l in obj.layers:
            open_ct[l] += int((gates[l][valid] > 0).sum())
    return {l: open_ct[l] / valid_ct for l in obj.layers}


# The frozen-gate_b fingerprint (psr_frozen_gate_b + check_psr_gates) may
# only RAISE when BOTH thresholds below hold, on top of the max_dead_frac
# fraction test. Each kills a measured false positive, while the real
# signatures -- gate_b frozen at 0.924214 across 18/32 (safety_refusal)
# and 11/32 (si_constitution) layers -- clear both with room to spare.
#
# FROZEN_LAYER_FLOOR -- minimum ABSOLUTE number of layers sharing the
# value. A fraction alone is not evidence on a small model: a 2-layer
# smoke fit put 2/2 == 100% of its layers into one group, which is
# unremarkable, whereas 6+ independently-trained gates landing identical
# to 6 decimals at a common departed value has no innocent mechanism.
# 6 is about half the smaller real signature (11/32), so both real cases
# fire with ~2x margin while 2/2 can never reach the floor -- and the
# floor doubles as the minimum object depth the fingerprint applies to at
# all (an object with < 6 layers can never fire it, so no separate
# min-total-layer knob is needed).
FROZEN_LAYER_FLOOR = 6

# FROZEN_DEPART_FLOOR -- minimum distance of the shared value from the
# 1.0 init (PSRObject gate_b inits to torch.ones). The freeze mechanism
# is one full-magnitude AdamW impulse then zeros forever; at the default
# lr 5e-3 that impulse response carries gate_b ~15 lr-units (m-hat /
# sqrt(v-hat) ~ 1 on the impulse step, and the beta1/beta2 decay tail
# multiplies it), measured 0.924214 = 0.075786 from init. Gates that have
# merely taken a couple of small optimizer steps stay within tiny
# fractions of an lr of init -- the smoke-fit false positive sat at
# 0.99995, 5e-5 (0.01 lr-units) from init. 0.01 = two full-magnitude
# default-lr steps: ~7.6x BELOW the measured freeze (any genuine
# departed-then-frozen value fires; nothing hard-codes 0.924214) and 200x
# ABOVE the observed near-init coincidence (barely-trained gates cannot
# fire, however many layers coincide).
FROZEN_DEPART_FLOOR = 0.01


def psr_frozen_gate_b(obj: PSRObject, *, decimals: int = 6,
                      min_departure: float = FROZEN_DEPART_FLOOR,
                      ) -> dict[float, list[int]]:
    """Frozen-parameter fingerprint, no forward pass needed: groups of
    layers whose ``gate_b`` values are IDENTICAL to ``decimals`` decimals.

    A gate that saturated shut on the first AdamW step receives one nonzero
    gradient and then zeros forever, so every such layer lands on the SAME
    ``gate_b`` — the saved psr_kl_all artifacts show 0.924214 identical to
    6 decimals across the frozen block of BOTH SIs (the impulse-response
    fit holds for gate_b; gate_w's frozen |w| is ~0.0286 and does NOT match
    the naive impulse simulation). Independently-trained live layers land
    on distinct values, so a multi-layer group at one non-init value is the
    freeze's signature. Unlike liveness this is a property of the
    PARAMETERS, not of a probe sample — immune to thin probes.

    Layers within ``min_departure`` of the init value (b == 1.0,
    default :data:`FROZEN_DEPART_FLOOR`) are EXCLUDED -- the whole
    near-init BAND, not just the exact init: an epochs=0 construction
    legitimately leaves every layer at 1.0, and a short/small fit
    legitimately leaves every layer NEAR it (a 2-layer smoke fit once
    left both gates at 0.99995 -- 5e-5 from init, gates that had barely
    moved -- and an exact-init-only exclusion called that a freeze). The
    real freeze sits at 0.924214, 0.0758 ~ 15 default-lr-units from init:
    a clear DEPARTURE (the AdamW impulse response) that the band cannot
    mask. Only groups of two or more layers are returned; the RAISE in
    :func:`check_psr_gates` additionally requires
    :data:`FROZEN_LAYER_FLOOR` of them."""
    init = 1.0   # PSRObject gate_b init (torch.ones)
    groups: dict[float, list[int]] = {}
    for l in obj.layers:
        v = round(float(obj.gate_b[l].detach()), decimals)
        groups.setdefault(v, []).append(l)
    return {v: ls for v, ls in sorted(groups.items())
            if len(ls) >= 2 and abs(v - init) >= min_departure}


def check_psr_gates(model, tok, *, obj: PSRObject,
                    examples: list[DistillExample],
                    max_len: int = 768,
                    max_dead_frac: float = 0.25,
                    tag: str = "psr",
                    min_probe: int = GUARD_PROBE_FLOOR) -> dict[int, float]:
    """Fail-loudly dead-gate guard: compute :func:`psr_gate_liveness` over
    EVERY supplied example, plus the forward-free frozen-``gate_b``
    fingerprint (:func:`psr_frozen_gate_b`), and raise
    :class:`DeadGateError` when MORE THAN ``max_dead_frac`` of the layers
    are dead (gate closed at every probe token), or when the fingerprint
    fires -- at least :data:`FROZEN_LAYER_FLOOR` layers, AND more than
    ``max_dead_frac`` of the object, sharing one ``gate_b`` identical to
    6 decimals that sits at least :data:`FROZEN_DEPART_FLOOR` from the
    1.0 init; anything smaller still warns, never passes silently.
    Returns the liveness dict so callers can log it.

    Call contract — ``obj`` and ``examples`` are KEYWORD-ONLY (a ported
    call once died twice guessing the order against ``load_object``'s
    path-first signature). What each argument must be:

    - ``model``, ``tok``: the loaded HF causal LM (already on its device)
      and its tokenizer — positional, like every sisteer API.
    - ``obj``: the trained/re-hydrated :class:`PSRObject`
      (``load_object(path, model=model, tok=tok)``).
    - ``examples``: the RENDERED teacher/student probe pairs
      (``pairs_to_distill_examples(tok, pairs, instruction)``) — ideally
      the pairs the object trained on. NOT the pairs file: on a cluster the
      pairs jsonl must live on SHARED storage (``$SHARED/...`` — cluster
      ``/home`` is node-local disk, invisible from compute nodes), and the
      path should be validated BEFORE the model loads.

    Compute-node usage — validate first, load second, guard last::

        pairs_path = validate_probe_pairs_path(
            "$SHARED/two_si/pairs/safety_refusal.jsonl")
        model, tok = load_model_and_tokenizer(...)          # expensive
        obj = load_object(object_path, model=model, tok=tok)
        check_psr_gates(model, tok, obj=obj,
                        examples=pairs_to_distill_examples(
                            tok, load_pairs(pairs_path), instruction))

    ``min_probe`` (default :data:`GUARD_PROBE_FLOOR` = 64): the guard
    REFUSES to rule on fewer examples — the >25%-dead criterion is a
    SAMPLE statistic from a probe forward, and a 4-example probe once
    yielded a monotone, two-object-corroborated result that vanished at
    n=32. Lower it EXPLICITLY only when ``examples`` is the COMPLETE
    training set of a deliberately small fit (the embedded guard in
    ``train_psr`` does exactly that): probing every token the fit ever
    trained on is exhaustive, not thin. The frozen-``gate_b`` fingerprint
    is checked either way and needs no forward, so it is immune to probe
    thinness.

    The 0.25 default is calibrated to the failure this guard exists to
    catch, not to noise: the diffmean-init all-layer KL PSR died at 18/32
    layers (~56%, safety_refusal) and 11/32 (~34%, si_constitution) — the
    bar fires on both — while one incidentally-closed layer out of 32
    (~3%) only warns. In THAT wave both psr_mse_all objects were pristine
    (0 frozen gates, 32/32 distinct gate_b), matching "the mse path's
    reg_coeff 0.1 gate_reg vs the KL path's nothing" — but the 23k-pair
    wave (2026-08-22) showed the regularizer alone does not scale: at
    5,165 steps/epoch psr_mse_all died at 26/32, 17/32 and 27/32 layers
    WITH the term (the ratchet: a closed ReLU silenced the term's own
    gradient), which is what forced the straight-through leak
    (GATE_REG_LEAK) and the mse loop's warmup/decay/clipping. For a
    single-layer PSR any dead gate is 100% dead and rightly fatal: the
    whole object is a no-op. (The protections reduce the failure's
    incidence, never this guard's authority.)"""
    examples = _as_probe_examples(examples)
    if len(examples) < min_probe:
        raise ValueError(
            f"{tag}: only {len(examples)} probe example(s) supplied but "
            f"the dead-gate verdict requires at least {min_probe} "
            f"(GUARD_PROBE_FLOOR={GUARD_PROBE_FLOOR}) — a save/no-save "
            "decision must not rest on a handful of examples (a 4-example "
            "probe once produced a monotone result, corroborated across "
            "two independently-trained objects, that vanished at n=32). "
            "Supply more pairs; pass min_probe=len(examples) explicitly "
            "ONLY when these examples are the complete training set of a "
            "deliberately small fit.")
    liveness = psr_gate_liveness(model, tok, obj=obj, examples=examples,
                                 max_len=max_len, n_probe=len(examples))
    dead = sorted(l for l, frac in liveness.items() if frac == 0.0)
    if dead and len(dead) / len(obj.layers) > max_dead_frac:
        raise DeadGateError(
            f"{tag}: {len(dead)}/{len(obj.layers)} PSR gate(s) DEAD "
            f"(ReLU(w.h + b) == 0 at every probe token) at layer(s) {dead}. "
            "A dead gate steers nothing and receives zero gradient — to its "
            "own (w, b) AND to that layer's direction — so it can never "
            "recover: ReLU's zero gradient silences the anti-dead-gate "
            "regularizer too (both objectives carry it since 2026-08-20; "
            "the KL path had none when the failure below was recorded). "
            "Refusing to treat this object as trained; do not save it. "
            "(This is how psr_kl_all once trained 966 steps ~30x worse than "
            "a no-op: 32 full-strength diffmean directions blew step-0 KL "
            "to 9.28 vs a 0.148 no-op floor and the first AdamW step drove "
            "deep-layer gates permanently shut — 18/32 layers on "
            "safety_refusal, 11/32 on si_constitution — gate_b frozen at "
            "0.924214.)")
    if dead:
        warnings.warn(
            f"{tag}: PSR gate(s) dead at layer(s) {dead} "
            f"({len(dead)}/{len(obj.layers)}, at or under the "
            f"max_dead_frac={max_dead_frac} bar) — those layers steer "
            "nothing and cannot recover (a fully closed ReLU gate gets "
            "zero gradient from the objective and the regularizer alike).",
            RuntimeWarning, stacklevel=2)
    frozen = psr_frozen_gate_b(obj)
    if frozen:
        frozen_layers = sorted(l for ls in frozen.values() for l in ls)
        detail = "; ".join(f"gate_b == {v} at layers {ls}"
                           for v, ls in frozen.items())
        if (len(frozen_layers) >= FROZEN_LAYER_FLOOR
                and len(frozen_layers) / len(obj.layers) > max_dead_frac):
            raise DeadGateError(
                f"{tag}: frozen-parameter fingerprint: "
                f"{len(frozen_layers)}/{len(obj.layers)} layers share an "
                f"identical non-init gate_b to 6 decimals ({detail}) — "
                f"at least FROZEN_DEPART_FLOOR={FROZEN_DEPART_FLOOR} from "
                f"the 1.0 init, across at least "
                f"FROZEN_LAYER_FLOOR={FROZEN_LAYER_FLOOR} layers and more "
                f"than max_dead_frac={max_dead_frac} of the object — "
                "AdamW's signature for a gate that got one gradient "
                "impulse and then zeros forever (the saved psr_kl_all "
                "artifacts froze at 0.924214, 0.0758 from init, across "
                "18/32 and 11/32 layers). This check reads the parameters "
                "directly, so it holds even where a probe sample could "
                "not. Refusing to treat this object as trained; do not "
                "save it.")
        warnings.warn(
            f"{tag}: layers sharing an identical non-init gate_b to 6 "
            f"decimals ({detail}) — under the raise bar (which needs at "
            f"least FROZEN_LAYER_FLOOR={FROZEN_LAYER_FLOOR} such layers "
            f"AND more than max_dead_frac={max_dead_frac} of the object), "
            "but this is the frozen-gate fingerprint; check those layers.",
            RuntimeWarning, stacklevel=2)
    return liveness


def _psr_step_loss(teacher_b, student_b, loss_layers, t_hidden, s_hidden,
                   pos_mask, t_p, reg, reg_coeff, w_mse: float = 1.0,
                   w_reg: float = 1.0):
    """The mse step loss, verbatim from the historical inline block (the
    mask→loss guard included) — factored out so the caller can run it,
    and the backward, INSIDE apply()/_capture_hidden (checkpointing
    recompute needs the hooks registered; see the call site).

    ``w_mse`` / ``w_reg`` are gradient-accumulation share weights (the
    micro-batch's completion-token share of its group for the mse — its
    per-layer mean divides by positions x dim, and dim is constant — and
    its ROW share for the regularizer, gate_regularizer's own
    denominator), so a group's summed contributions equal one true
    batch's loss. 1.0 (the default) skips the multiplies: byte-identical
    to the historical step."""
    t_mask = (completion_token_mask_suffix(teacher_b, t_p)
              if t_p >= 1 else teacher_b.labels != -100)
    s_mask = student_b.labels != -100
    if pos_mask is not None and bool((s_mask & ~pos_mask).any()):
        # The mse loss reads ACTIVATIONS at completion-token
        # positions; under the header-steered boundary
        # (prompt_len - H <= prompt_len) every completion position
        # is steered by construction, so a completion position
        # outside the hook's own mask means the boundary was
        # misderived — refuse to train on the wrong support (the
        # activation-side twin of loss_gather_masks' guard).
        raise ValueError(
            "mask→loss wiring (psr mse): a supervised completion "
            "position falls OUTSIDE the steering mask — the "
            "boundary was misderived (padded batch width instead "
            "of each row's prompt_len minus H?)")
    mse = torch.stack([
        F.mse_loss(s_hidden[l][s_mask].float(),
                   t_hidden[l][t_mask].float())
        for l in loss_layers
    ]).sum()  # SUM over layers (their reduction), mean within
    step_loss = (mse if w_mse == 1.0 else w_mse * mse) + reg_coeff * (
        reg if w_reg == 1.0 else w_reg * reg)
    return step_loss, mse


def train_psr(
    model,
    tok,
    obj: PSRObject,
    examples: list[DistillExample],
    *,
    loss: Literal["mse", "kl"] = "mse",
    lr: float = 5e-3,
    weight_decay: float = 0.01,
    epochs: int = 6,
    batch_size: int = 2,
    max_len: int = 768,
    temperature: float = 1.0,
    reg_coeff: float = 0.1,
    seed: int = 0,
    log_fn=None,
    positions: str = "all",
    response_header_len: int | None = None,
    lr_schedule: Literal["warmup_cosine", "constant"] | None = None,
    warmup_steps: int | None = None,
    max_grad_norm: float | None = None,
    accum_steps: int = 1,
    teacher_prefix_cache: bool = False,
    probe_cb: Callable[[int, object], None] | None = None,
    probe_every: int | None = None,
    checkpoint_every: int | None = None,
    checkpoint_dir=None,
    checkpoint_key: str | None = None,
    divergence_guard=None,
    heldout: list[DistillExample] | None = None,
    early_stop: EarlyStopSpec | None = None,
    diagnostics: dict | None = None,
) -> list[float]:
    """Train the PSR gate and direction. Returns the loss history; the
    object is trained in place.

    ``teacher_prefix_cache``: the SI-prompted
    teacher pass — activations here, not logits — shares the SI's rendered
    prefix across every batch like the KL arms' teacher legs, so under the
    flag the prefix runs through the model ONCE and each step's teacher
    capture forwards only the suffix (:mod:`.prefix_cache`), with the
    completion-position mask re-based to suffix coordinates through the
    same guarded helper (an off-by-P on an ACTIVATION mask is the same
    silent-wrong-support failure as on a logit gather). The KL path
    forwards the flag to train_object_kl. Semantically exact,
    fp-tolerance-only difference; default ``False`` keeps other callers
    byte-identical — the two-SI config flag is the wave's switch.

    ``positions``/``response_header_len`` select the fit's mask leg: under
    ``"response"`` the PSR edit applies only from the per-row boundary on
    (header-steered when ``response_header_len`` is set; H is derived ONCE
    by the caller, never here). The kl arms inherit the full mask→loss
    wiring from train_object_kl; the mse arm's loss positions (completion
    tokens) sit inside the mask by construction — guarded, not assumed —
    and the anti-dead-gate regularizer pools gate mass over REAL ∩ STEERED
    positions only, since a gate alive only where the edit never applies
    steers nothing.

    mse (the paper-faithful arm): sum over psr_loss_layers (intervention
    layer .. last layer) of || h_steered(no-SI student) - h(SI teacher) ||^2
    on completion-token positions (mean within a layer over positions x
    dim, mean over the pooled batch rows), plus reg_coeff * the
    anti-dead-gate regularizer (their code-only value 0.1). Trains every
    param in obj.trainable — the paper trains the direction under MSE, so
    the registry constructs the mse arm with train_direction=True
    (mean-diff kept as INIT only, a documented init-only deviation).
    kl: delegates to train_object_kl (identical objective/schedule to
    every other KL-trained arm — a deliberate non-faithful variant),
    passing ``reg_coeff`` through so the SAME anti-dead-gate regularizer
    applies at the same 0.1 default (the 2026-08-20 confound fix: both
    objectives now carry both protective treatments, so an MSE-vs-KL
    difference isolates the objective).

    Optimizer schedule (mse loop ONLY — the 23k-pair wave fix, module
    docstring): ``lr_schedule=None`` resolves to ``"warmup_cosine"`` —
    linear warmup over ``warmup_steps`` (None = min(500, total//10):
    ~one-tenth of the wave's epoch 0, and ~1 step on a CPU-sized fit)
    then cosine decay to 0 — and ``max_grad_norm=None`` resolves to 1.0
    (``math.inf`` disables clipping explicitly). Warmup removes the
    step-1 AdamW impulse that blew the wave's mse loss up 85x before v
    was calibrated; decay anneals the fixed-lr noise floor the wave
    plateaued at; clipping caps the 1e8-scale spike steps that poisoned
    the second moment. ``lr_schedule="constant"`` reproduces the
    historical fixed-lr loop exactly. The KL path's optimizer loop lives
    in train_object_kl (another lane), so these three arguments are
    REFUSED there rather than silently dropped — pass them only with
    loss="mse".

    ``accum_steps`` (mse loop ONLY — the KL path refuses it like the
    schedule kwargs; the production KL arms accumulate through
    train_lockstep): gradient accumulation, effective batch =
    ``batch_size x accum_steps``. Each optimizer step spans
    ``accum_steps`` consecutive micro-batches of the shuffled order (the
    ragged final group is its own smaller step); each micro-batch's mse
    is weighted by its COMPLETION-TOKEN share of the group and the gate
    regularizer by its ROW share — each part's own loss denominator, so
    the accumulated gradient EQUALS one true batch's (a flat
    1/accum_steps would silently bias both whenever the group is
    ragged). Clipping fires ONCE per optimizer step on the ACCUMULATED
    gradient — never per micro-batch, which would silently keep
    micro-batch-level clipping granularity inside an effective-batch
    run. The lr schedule ticks and resolves warmup against OPTIMIZER
    steps (``epochs x ceil(n_batches / accum_steps)``); history, logs
    and the probe seam fire once per optimizer step. ``accum_steps=1``
    (the default) is byte-identical to the historical per-batch loop.

    ``probe_cb``/``probe_every`` — the trainer-level probe seam (the
    bidpo convention, closing review finding F7: psr_mse_all had NO
    mid-fit monitoring, so a gate collapse or wrong-direction drift could
    consume an entire long-context fit before the post-fit guards ran).
    ``probe_cb(step, obj)`` is invoked right after the optimizer step at
    every step where ``step % probe_every == 0`` (1-based optimizer
    steps). Unlike bidpo's transient clone, ``obj`` is the LIVE
    :class:`PSRObject` being fit in place — the monitor reads its
    ``directions``/``gate_w``/``gate_b`` for norm groups, runs
    ``obj.apply(model)`` for the dual-KL probe and true gate activity
    (fitmon dispatches on the object, never the arm name), and must
    treat it as read-only. The callback owns its own cadence/cost
    decisions; this loop adds no CPU<->GPU syncs on its behalf. The two
    parameters come together: a callback with no cadence (or a cadence
    with no callback) is refused rather than left silently inert. BOTH
    losses carry the seam: the mse loop invokes it directly; the KL path
    rides :func:`train_object_kl`'s per-step ``log_fn`` event (invoked
    exactly once per optimizer step, after ``opt.step()``, with the
    1-based step — step 0 is the startup record, never probed), so the
    contract is identical either way. A ``fitmon.TrainingDiverged``
    raised by the callback propagates out of this function — the fit
    aborts, the post-fit guards never run, and no history is returned,
    so a diverged object cannot be saved as trained.

    ``checkpoint_every``/``checkpoint_dir``/``checkpoint_key`` — the
    mid-fit durability seam (mse loop ONLY; the KL path refuses them like
    the schedule kwargs — its optimizer loop lives in train_object_kl,
    and the production KL arms checkpoint through train_lockstep's own
    seam). Default OFF: all three ``None`` leaves this loop byte-identical
    to before the seam existed. When on, the FULL trainer state (object
    trainables, optimizer state, history/step — the lr-schedule position —
    the epoch's shuffled order + position + ``random`` RNG state, torch
    CPU/CUDA RNG states) is written atomically every ``checkpoint_every``
    optimizer steps — always at a group end, so no partial accumulation
    state exists to carry — and on start the newest checkpoint whose
    ``checkpoint_key`` matches (the driver passes the cell's fit
    fingerprint) is restored EXACTLY; a mismatched or unreadable file is
    ignored loudly (fresh start). See :mod:`.fitckpt`;
    tests/test_fit_checkpoint.py pins interrupt/resume parity bitwise on
    CPU fp32, across an epoch boundary, under an active clip, on ragged
    batches.

    ``divergence_guard`` — the STEP-level divergence abort (mse loop
    ONLY; the KL path refuses an explicit setting — its optimizer loop
    lives in train_object_kl, and the production KL arms are guarded
    through train_lockstep's own seam). Default ON (``None``/``True`` =
    the production :class:`fitmon.StepDivergenceSpec`; ``False`` = off; a
    spec instance = custom tuning): per-epoch early stopping rules once
    per epoch (~8h on the slow fits), while an exploding run is visible
    in its step losses within minutes. The guard reads the loss float
    each optimizer step already synced (zero new syncs, observational —
    a healthy fit is bit-identical guard-on vs off) and raises
    :class:`fitmon.TrainingDiverged` (message ``step-divergence: ...``)
    on a verdict — the probe_cb abort contract: the fit unwinds, the
    post-fit guards never run, no history returns, the object is never
    treated as trained; HPO trials publish a terminal ``nan_diverged``
    summary and exit cleanly. tests/test_divergence_guard.py pins both
    directions.

    ``heldout`` + ``early_stop`` — per-epoch held-out early stopping (mse
    loop ONLY; the KL path refuses them like the schedule kwargs — the
    production KL psr arms early-stop through train_lockstep's own
    seam). psr_mse_all was the last arm running a FIXED epoch budget to
    the cap while every other arm stopped on a held-out curve. With
    ``heldout`` set, ``epochs`` becomes a CAP: after every epoch the SAME
    masked MSE as the training steps (:func:`_psr_step_loss`, mask→loss
    guard included) is evaluated on the held-out examples — teacher and
    steered student both under ``torch.no_grad`` (obj.apply is a forward
    hook already run grad-free by check_psr_gates and fitmon's probes),
    WITHOUT the gate regularizer (a protective TRAINING term must not
    decide which epoch counts as best — train_lockstep's held-out rule),
    token-weight-pooled over batches — and fed to a
    :class:`distill.EarlyStopper` (``early_stop``, or the default
    :class:`distill.EarlyStopSpec` when only ``heldout`` is given):
    training stops after ``patience`` non-improving epochs and the BEST
    epoch's trainables are restored — also when the cap is reached with a
    better epoch inside the patience window. A pre-training baseline
    point rides the curve only (never a restore target: a fit that only
    got worse must fail the gate/floor guards loudly, not restore its
    init). The held-out pass never reads the teacher prefix cache — a
    decision-bearing number stays AUTHORITATIVE AND UNCACHED
    (train_steered_lm's rule). The stopper's full state (best value,
    patience counter, best-object snapshot) and the curve ride the
    mid-fit checkpoint, so an interrupted fit resumes bitwise-exact —
    pinned by tests/test_fit_checkpoint.py. ``early_stop`` without
    ``heldout`` is refused (a silently inert stopper is this file's
    probe_cb-without-cadence bug class). ``diagnostics`` (optional dict,
    mse loop only) receives ``epochs_run`` / ``stopped_early`` /
    ``heldout_curve`` / ``best_epoch`` / ``best_heldout_loss`` — the
    train_steered_lm record keys — for the caller's fit report.
    """
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
    if loss in ("kl", "kl_hard"):
        if (lr_schedule, warmup_steps, max_grad_norm) != (None, None, None):
            raise ValueError(
                "lr_schedule/warmup_steps/max_grad_norm govern train_psr's "
                "own mse loop; the KL path delegates its optimizer loop to "
                "train_object_kl, which owns that schedule — refusing to "
                "silently drop them.")
        if int(accum_steps) != 1:
            raise ValueError(
                "accum_steps governs train_psr's own mse loop; the KL path "
                "delegates its optimizer loop to train_object_kl (and the "
                "production KL arms accumulate through train_lockstep's "
                "LockstepArmParams.accum_steps) — refusing to silently "
                "drop it.")
        if (checkpoint_every, checkpoint_dir, checkpoint_key) != (None,
                                                                  None,
                                                                  None):
            raise ValueError(
                "checkpoint_every/checkpoint_dir/checkpoint_key govern "
                "train_psr's own mse loop; the KL path delegates its "
                "optimizer loop to train_object_kl (and the production KL "
                "arms checkpoint through train_lockstep's seam) — refusing "
                "to silently drop a durability request.")
        if divergence_guard is not None:
            raise ValueError(
                "divergence_guard governs train_psr's own mse loop; the KL "
                "path delegates its optimizer loop to train_object_kl (and "
                "the production KL arms are step-guarded through "
                "train_lockstep) — refusing to silently drop an explicit "
                "divergence-guard setting.")
        if (heldout, early_stop, diagnostics) != (None, None, None):
            raise ValueError(
                "heldout/early_stop/diagnostics govern train_psr's own mse "
                "loop; the KL path delegates its optimizer loop to "
                "train_object_kl (and the production KL arms early-stop "
                "through train_lockstep's own heldout seam) — refusing to "
                "silently run the fixed epoch budget against an early-stop "
                "request.")
        from .objects import train_object_kl

        kl_log_fn = log_fn
        if probe_cb is not None:
            # The KL path's optimizer loop lives in train_object_kl
            # (another lane), whose per-step log_fn call IS the exact
            # per-optimizer-step event this seam needs: invoked once per
            # step, right after opt.step(), with the 1-based step ordinal
            # (step 0 is startup_loss_check's record, not an optimizer
            # step — excluded). Riding it gives psr_kl the identical
            # probe_cb(step, obj) contract as the mse loop below without
            # a second seam; obj is trained in place, so the closure's
            # obj IS the live object. Pinned by
            # tests/test_psr.py::test_probe_cb_fires_on_the_kl_path.
            def kl_log_fn(record, step, _inner=log_fn):
                if _inner:
                    _inner(record, step)
                if step >= 1 and step % probe_every == 0:
                    probe_cb(step, obj)

        history = train_object_kl(
            model, tok, obj, examples,
            lr=lr, weight_decay=weight_decay, epochs=epochs,
            batch_size=batch_size, max_len=max_len,
            temperature=temperature, seed=seed, log_fn=kl_log_fn,
            loss=("hard" if loss == "kl_hard" else "soft"),
            positions=positions, response_header_len=response_header_len,
            reg_coeff=reg_coeff,
            teacher_prefix_cache=teacher_prefix_cache,
        )
        # Fail-loudly dead-gate guard. The KL losses now carry the same
        # anti-dead-gate regularizer as the mse path (the 2026-08-20
        # confound fix), which makes gate death LESS LIKELY — but a gate
        # that closes anyway is still unrecoverable (ReLU's zero gradient
        # silences the regularizer too), so the guard stays, unweakened:
        # refuse to hand such an object back as "trained" (psr_kl_all,
        # then regularizer-less, once trained 966 steps ~30x worse than a
        # no-op with 18/32 resp. 11/32 dead layers, silently).
        # min_probe: the training set IS the probe pool here — the guard
        # runs exhaustively over every example the fit trained on, so a
        # set smaller than GUARD_PROBE_FLOOR is complete, not thin; lower
        # the floor to the population rather than refusing a deliberately
        # small fit (the CPU test fixtures train on a handful of pairs).
        check_psr_gates(model, tok, obj=obj, examples=examples,
                        max_len=max_len, tag=f"psr_{loss}",
                        min_probe=min(GUARD_PROBE_FLOOR, len(examples)))
        return history
    if loss != "mse":
        raise ValueError("loss must be 'mse', 'kl', or 'kl_hard'")
    if positions != "all":
        obj.set_fit_positions(positions, response_header_len)

    accum_steps = int(accum_steps)
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps!r}")
    if early_stop is not None and not heldout:
        raise ValueError(
            "early_stop without heldout — the stopper would have no "
            "held-out curve to rule on and the fit would silently run the "
            "full epoch budget wearing an early-stop request (the "
            "probe_cb-without-cadence bug class); pass the held-out "
            "examples or drop the spec.")
    checkpoint_every = resolve_checkpoint_kwargs(
        checkpoint_every, checkpoint_dir, checkpoint_key)
    from .fitmon import resolve_divergence_guard

    divguard = resolve_divergence_guard(divergence_guard, tag="psr_mse")
    if lr_schedule is None:
        lr_schedule = "warmup_cosine"
    if max_grad_norm is None:
        max_grad_norm = 1.0
    # Echo the EFFECTIVE optimiser settings, after resolution. The caller
    # cannot compute these: `max_grad_norm=None` means "clip at 1.0" here,
    # not "no clipping" (only math.inf disables it), and `lr_schedule=None`
    # means warmup_cosine. A driver-side banner reports what it PASSED; only
    # this point knows what will run.
    print(f"[psr-optimizer] loss={loss} lr={lr} schedule={lr_schedule} "
          f"warmup_steps={warmup_steps} max_grad_norm={max_grad_norm} "
          f"(clipping {'OFF' if not math.isfinite(max_grad_norm) else 'ON'})"
          f" accum_steps={accum_steps} checkpoint_every={checkpoint_every}",
          flush=True)

    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    obj.to_(device)
    # weight_decay exposed 2026-08-27 (re-tune): the default preserves
    # AdamW's own 0.01, which every prior fit silently trained under.
    opt = torch.optim.AdamW(obj.trainable, lr=lr, weight_decay=weight_decay)
    loss_layers = psr_loss_layers(obj, model)

    _prog = TokenizeProgress(len(examples), label=f"psr_{loss} train",
                             max_len=max_len)
    encoded = []
    for _i, ex in enumerate(examples, 1):
        encoded.append(_encode_aligned(tok, ex, max_len))
        _prog.tick(_i)
    _prog.done()
    batches: list[tuple[Batch, Batch]] = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        batches.append(
            (collate([t for t, _ in chunk], tok.pad_token_id).to(device),
             collate([s for _, s in chunk], tok.pad_token_id).to(device))
        )

    # Teacher prefix cache for the per-step activation capture (see the
    # docstring) — one prefix forward for the whole fit, replayed each step.
    tpc = (prefix_cache_for_batches(model, [t for t, _ in batches])
           if teacher_prefix_cache else None)

    # OPTIMIZER-step budget: with gradient accumulation an optimizer step
    # spans accum_steps micro-batches (the ragged final group is its own
    # smaller step), and the schedule ticks/resolves against optimizer
    # steps — at accum_steps=1 this is exactly the historical
    # epochs * n_batches.
    total_steps = epochs * math.ceil(len(batches) / accum_steps)
    if warmup_steps is None:
        warmup_steps = min(500, max(1, total_steps // 10))
    # Accumulation share weights' denominators, known before any forward:
    # completion-token count (the mse mean's position denominator; dim is
    # constant) and row count (gate_regularizer's row-mean denominator).
    n_tok = [int((sb.labels != -100).sum()) for _tb, sb in batches]
    n_rows = [int(sb.input_ids.shape[0]) for _tb, sb in batches]

    # Held-out batches for the per-epoch early-stop pass (docstring): the
    # fit's exact encoding, never shuffled, never trained on.
    h_batches: list[tuple[Batch, Batch]] = []
    if heldout:
        _hprog = TokenizeProgress(len(heldout), label=f"psr_{loss} held-out",
                                  max_len=max_len)
        h_encoded = []
        for _i, ex in enumerate(heldout, 1):
            h_encoded.append(_encode_aligned(tok, ex, max_len))
            _hprog.tick(_i)
        _hprog.done()
        for i in range(0, len(h_encoded), batch_size):
            chunk = h_encoded[i : i + batch_size]
            h_batches.append(
                (collate([t for t, _ in chunk], tok.pad_token_id).to(device),
                 collate([s for _, s in chunk], tok.pad_token_id).to(device)))
    h_n_tok = [int((sb.labels != -100).sum()) for _tb, sb in h_batches]

    def _heldout_mse() -> float:
        """Token-weighted mean held-out MSE (object active, no grad): the
        exact :func:`_psr_step_loss` masked MSE — the mask→loss guard
        included — WITHOUT the gate regularizer (a protective TRAINING
        term must not decide which epoch counts as best; train_lockstep's
        held-out rule) and WITHOUT a backward. The teacher leg is the
        RESIDENT model's plain full forward, deliberately never the
        prefix cache: this number selects the restored epoch, and a
        decision-bearing value stays AUTHORITATIVE AND UNCACHED
        (train_steered_lm's rule) — which also keeps t_p at 0, full-width
        masks on both sides."""
        tot = toks = 0.0
        with torch.no_grad():
            for hbi, (teacher_b, student_b) in enumerate(h_batches):
                with _capture_hidden(model, loss_layers) as t_cap:
                    model(input_ids=teacher_b.input_ids,
                          attention_mask=teacher_b.attention_mask)
                    t_hidden = {l: t_cap[l].detach() for l in loss_layers}
                pos_mask = None
                with obj.apply(model, scale=1.0) as mask_handle, \
                        _capture_hidden(model, loss_layers) as s_cap:
                    if mask_handle is not None:
                        # Per-row boundaries BEFORE the forward — the
                        # training loop's exact convention.
                        mask_handle.set_prompt_lens(steer_prompt_boundaries(
                            student_b, mask_handle.response_header_len))
                    model(input_ids=student_b.input_ids,
                          attention_mask=student_b.attention_mask)
                    s_hidden = dict(s_cap)
                    if mask_handle is not None:
                        b_, s_ = student_b.input_ids.shape
                        pos_mask = mask_handle._position_mask(
                            b_, s_, student_b.input_ids.device).bool()
                    _sl, mse = _psr_step_loss(
                        teacher_b, student_b, loss_layers, t_hidden,
                        s_hidden, pos_mask, 0, 0.0, 0.0)
                # Each batch's mse is a per-layer mean over its OWN
                # completion tokens (x dim, constant) summed over layers;
                # token-weighting makes the pool equal the metric over the
                # pooled set (lockstep's _heldout_pass arithmetic).
                tot += float(mse.detach()) * h_n_tok[hbi]
                toks += h_n_tok[hbi]
        return tot / max(toks, 1)

    stopper: EarlyStopper | None = (
        EarlyStopper(early_stop or EarlyStopSpec()) if h_batches else None)
    heldout_curve: list[float] = []

    def _lr_at(i: int) -> float:
        """The schedule's lr for 0-based step ``i`` of ``total_steps``."""
        if lr_schedule == "constant":
            return lr
        if i < warmup_steps:
            return lr * (i + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return lr
        frac = (i - warmup_steps) / (total_steps - warmup_steps)
        return lr * 0.5 * (1.0 + math.cos(math.pi * frac))

    history: list[float] = []
    step = 0
    # -- mid-fit checkpoint seam (default OFF — see the docstring) -----------
    start_epoch = 0
    resume_order: list[int] | None = None
    resume_batches_done = 0
    ckpt_last_step = 0
    if checkpoint_every is not None:
        ck = load_checkpoint(
            checkpoint_dir, checkpoint_key,
            expect={"trainer": "train_psr_mse",
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
            start_epoch = int(ck["epoch"])
            resume_order = [int(b) for b in ck["order"]]
            resume_batches_done = int(ck["batches_done"])
            ckpt_last_step = step
            # Early-stop state travels WITH the trainer state: the best
            # value, the patience counter, and the best-object snapshot —
            # a stopper rebuilt fresh at resume could only ever restore a
            # post-resume epoch, silently shifting the kept weights
            # (train_lockstep's checkpointed-stopper rule).
            heldout_curve = [float(v) for v in ck.get("heldout_curve") or []]
            st = ck.get("stopper")
            if stopper is not None and st is not None:
                stopper.best = st["best"]
                stopper.best_epoch = st["best_epoch"]
                stopper.epochs_seen = st["epochs_seen"]
                stopper._since_best = st["since_best"]
                stopper._snapshot = st["snapshot"]
            print(f"[fitckpt] train_psr(mse) RESUMING from checkpoint: "
                  f"epoch {start_epoch}, micro-batch "
                  f"{resume_batches_done}/{len(batches)} in epoch, "
                  f"optimizer step {step} — skipping to the checkpointed "
                  "position", flush=True)
    if stopper is not None and not heldout_curve:
        # Pre-training baseline point (curve only — never a stopper update,
        # so the init can never become the restore target; a fit that only
        # got worse must fail the gate/floor guards loudly instead —
        # train_lockstep's epoch -1 rule). A mid-fit resume restored the
        # original curve above and skips this.
        heldout_curve.append(_heldout_mse())
        if log_fn:
            log_fn({"psr/heldout_loss_init": heldout_curve[-1]}, 0)
    epochs_run = start_epoch
    stopped_early = False
    for epoch in range(start_epoch, epochs):
        if resume_order is not None and epoch == start_epoch:
            # Mid-epoch resume: the checkpoint carries this epoch's
            # already-shuffled order and the consumed micro-batch count —
            # always a group boundary (checkpoints fire only at group
            # ends), so the range below restarts on the next group.
            order = resume_order
            skip = resume_batches_done
        else:
            order = list(range(len(batches)))
            rng.shuffle(order)
            skip = 0
        # Gradient accumulation groups along the shuffled order (bidpo's
        # ceil-path shape): one optimizer step per group; the ragged final
        # group normalizes over its own members. accum_steps=1 makes every
        # batch its own group — the historical per-batch loop, op for op.
        for g0 in range(skip, len(order), accum_steps):
            group = order[g0:g0 + accum_steps]
            grp_tok = sum(n_tok[b] for b in group)
            grp_rows = sum(n_rows[b] for b in group)
            acc_loss = acc_mse = acc_reg = 0.0
            for k, bi in enumerate(group):
                teacher_b, student_b = batches[bi]
                # Each part weighted by its OWN loss denominator (see
                # _psr_step_loss): the accumulated gradient equals one
                # true batch of the group's rows. 1.0 at accum_steps=1.
                w_mse = 1.0 if accum_steps == 1 else n_tok[bi] / grp_tok
                w_reg = 1.0 if accum_steps == 1 else n_rows[bi] / grp_rows
                # Teacher activations are recomputed PER BATCH inside the
                # loop rather than cached up front: with the S-PSR
                # objective scoring every layer from the intervention to
                # the top, an up-front cache is
                # O(n_batches x n_layers x B x S x d) fp32 — tens of GB at
                # 9B scale for the all-layer arm — while recomputing costs
                # one extra no-grad forward per step (cheap next to the
                # student's backward). Memory-flat wins on shared GPUs.
                t_p = tpc.match_len(teacher_b) if tpc is not None else 0
                with torch.no_grad(), \
                        _capture_hidden(model, loss_layers) as t_cap:
                    if t_p >= 1:
                        # Cached teacher pass: the hooks capture
                        # SUFFIX-width hidden states (positions t_p..S-1);
                        # the completion mask below is re-based to match,
                        # through the guarded suffix helper — never by
                        # hand.
                        tpc.forward_ids(model, teacher_b.input_ids,
                                        teacher_b.attention_mask, t_p)
                    else:
                        model(input_ids=teacher_b.input_ids,
                              attention_mask=teacher_b.attention_mask)
                    t_hidden = {l: t_cap[l].detach() for l in loss_layers}
                pos_mask = None
                with obj.apply(model, scale=1.0) as mask_handle, \
                        _capture_hidden(model, loss_layers) as s_cap:
                    if mask_handle is not None:
                        # Per-row boundaries BEFORE the forward (never
                        # derived from the padded batch width — the trap).
                        mask_handle.set_prompt_lens(steer_prompt_boundaries(
                            student_b, mask_handle.response_header_len))
                    model(input_ids=student_b.input_ids,
                          attention_mask=student_b.attention_mask)
                    s_hidden = dict(s_cap)
                    # The shared regularizer assembly (REAL ∩ STEERED
                    # pooling) must run inside apply() — the gate stash
                    # clears at exit.
                    reg = obj.gate_reg_term(student_b.attention_mask,
                                            mask_handle)
                    if mask_handle is not None:
                        b_, s_ = student_b.input_ids.shape
                        pos_mask = mask_handle._position_mask(
                            b_, s_, student_b.input_ids.device).bool()
                    # Loss + BACKWARD stay INSIDE apply()/_capture_hidden:
                    # under activation checkpointing the backward
                    # RECOMPUTES each layer's forward, and a recompute
                    # without the steering hooks is a different graph —
                    # torch's non-reentrant checkpoint refuses it
                    # (diagnosed op-by-op 2026-08-27: the recompute stream
                    # was missing exactly the hook's ops; the historical
                    # exit-then-backward order was the cause). Math-neutral
                    # without checkpointing: hooks only shape the forward
                    # graph, and traversing an already-built graph is
                    # placement-independent (tests/test_ckpt_grad_parity.py
                    # pins grad parity).
                    step_loss, mse = _psr_step_loss(
                        teacher_b, student_b, loss_layers, t_hidden,
                        s_hidden, pos_mask, t_p, reg, reg_coeff,
                        w_mse=w_mse, w_reg=w_reg)
                    if k == 0:
                        opt.zero_grad()
                    step_loss.backward()
                # Completion-token positions; identical token sequences on
                # both sides, so the selected counts match. Under the
                # cached teacher pass the teacher hidden states are
                # suffix-width, so the teacher mask is re-based by t_p via
                # the guarded suffix helper (it refuses if any completion
                # token sits inside the prefix — the off-by-P trap on the
                # activation side); the STUDENT side is untouched
                # (full-width forward, gradients).
                acc_loss += float(step_loss.detach())
                acc_mse += (float(mse.detach()) if w_mse == 1.0
                            else float(mse.detach()) * w_mse)
                acc_reg += (float(reg.detach()) if w_reg == 1.0
                            else float(reg.detach()) * w_reg)
            # Group end: ONE clip, on the ACCUMULATED gradient — never per
            # micro-batch, which would silently reproduce micro-batch-level
            # clipping inside an effective-batch run.
            grad_norm_t = None
            if math.isfinite(max_grad_norm):
                # clip_grad_norm_ returns the PRE-clip global norm — the
                # divergence signal a monitor watches (bidpo/grad_norm's
                # convention); the optimizer consumes the clipped grads.
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    obj.trainable, max_grad_norm)
            elif log_fn:
                # Clipping explicitly disabled (math.inf): compute the
                # norm for telemetry only — F7 named the pre-clip grad
                # norm a deliverable, clip setting or no clip setting.
                grads = [p.grad.detach().float().norm()
                         for p in obj.trainable if p.grad is not None]
                if grads:
                    grad_norm_t = torch.norm(torch.stack(grads))
            cur_lr = _lr_at(step)
            for grp in opt.param_groups:
                grp["lr"] = cur_lr
            opt.step()
            step += 1
            history.append(acc_loss)
            if log_fn:
                rec = {"psr/mse_loss": acc_loss,
                       "psr/mse": acc_mse,
                       "psr/gate_reg": acc_reg,
                       "psr/lr": cur_lr,
                       "epoch": epoch}
                if grad_norm_t is not None:
                    rec["psr/grad_norm"] = float(grad_norm_t)
                log_fn(rec, step)
            if divguard is not None:
                # Step-level divergence abort (docstring): rules on the
                # loss float this step already synced; TrainingDiverged
                # propagates — the documented probe_cb abort contract —
                # so a diverging fit dies in steps, not at the epoch
                # boundary.
                divguard.step(step, acc_loss)
            if probe_cb is not None and step % probe_every == 0:
                # The probe seam (docstring): the LIVE object, right after
                # the optimizer step. The callback gates its own cost; a
                # fitmon.TrainingDiverged raised here propagates and
                # aborts the fit before the object can be treated as
                # trained.
                probe_cb(step, obj)
            if (checkpoint_every is not None
                    and step - ckpt_last_step >= checkpoint_every):
                # Group end == optimizer-step boundary: no partial
                # accumulation state exists here. GPU->CPU serialization
                # happens only at these moments.
                stopper_state = None
                if stopper is not None:
                    # The early-stop state, whole (train_lockstep's
                    # _save_ckpt shape): only the CHECKPOINTED snapshot can
                    # restore a best epoch that predates the resume point.
                    stopper_state = {
                        "best": stopper.best,
                        "best_epoch": stopper.best_epoch,
                        "epochs_seen": stopper.epochs_seen,
                        "since_best": stopper._since_best,
                        "snapshot": (None if stopper._snapshot is None else
                                     [t.detach().cpu().clone()
                                      for t in stopper._snapshot]),
                    }
                save_checkpoint(checkpoint_dir, {
                    "checkpoint_key": checkpoint_key,
                    "trainer": "train_psr_mse",
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
                    "stopper": stopper_state,
                })
                ckpt_last_step = step
                print(f"[fitckpt] train_psr(mse) checkpoint written: "
                      f"epoch {epoch}, micro-batch {g0 + len(group)}/"
                      f"{len(batches)} in epoch, optimizer step {step}",
                      flush=True)
        epochs_run = epoch + 1
        if stopper is not None:
            # Per-epoch held-out ruling (docstring): the same masked MSE,
            # regularizer excluded, no grad — the stopper snapshots a new
            # best and stops after `patience` non-improving epochs.
            hl = _heldout_mse()
            heldout_curve.append(hl)
            if log_fn:
                log_fn({"psr/heldout_loss": hl, "epoch": epoch}, step)
            if stopper.update(hl, obj.trainable):
                stopped_early = True
                if log_fn:
                    log_fn({"psr/stopped_epoch": epoch + 1,
                            "psr/best_epoch": stopper.best_epoch}, step)
                break
    if stopper is not None:
        # A fit that reached the cap with a better epoch inside the
        # patience window keeps the BEST epoch's weights too, exactly like
        # a stopped fit (train_lockstep's restore-at-cap rule); the gate
        # guard below rules on the weights that would actually ship.
        stopper.restore_best(obj.trainable)
    if diagnostics is not None:
        diagnostics.update({
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "heldout_curve": list(heldout_curve),
            "best_epoch": stopper.best_epoch if stopper else None,
            "best_heldout_loss": stopper.best if stopper else None,
        })
    # The mse objective's gate regularizer makes gate death unlikely — but
    # if it happens anyway, fail as loudly as the KL path does rather than
    # handing back an object with do-nothing layers. min_probe: exhaustive
    # over the full training set, see the KL branch's note.
    check_psr_gates(model, tok, obj=obj, examples=examples, max_len=max_len,
                    tag="psr_mse",
                    min_probe=min(GUARD_PROBE_FLOOR, len(examples)))
    return history


# --------------------------------------------------------------------------- #
# Import-time guard: the hardened dead-gate guard must be defined exactly once  #
# --------------------------------------------------------------------------- #
# check_psr_gates is the fail-loudly heart of the PSR arms (the probe floor +
# the frozen-parameter fingerprint). A second, older copy of it -- the kind an
# auto-merge that re-copies a block can introduce -- would silently REBIND the
# name to whichever ran last, reverting the hardening while every import still
# succeeds. Make that a LOUD ImportError at module load (a name collision, not
# a silent shadow). Chosen over an ``__all__`` because ``__all__`` governs only
# ``import *`` and would not stop a rebinding shadow at all; nothing does
# ``from ...psr import *``. tests/test_psr.py mirrors this offline so CI catches
# a duplicate without importing torch. The count MUST stay 1 -- do not let any
# other line in this file carry the exact target substring.
def _assert_single_gate_definition() -> None:
    marker = "def " "check_psr_gates"   # split so THIS line never self-matches
    n = Path(__file__).read_text().count(marker)
    if n != 1:
        raise ImportError(
            f"psr.py defines the dead-gate guard {n} times (expected 1): a "
            "duplicate would silently shadow the hardened version (probe floor "
            "+ frozen-parameter fingerprint). Remove the duplicate.")


_assert_single_gate_definition()
