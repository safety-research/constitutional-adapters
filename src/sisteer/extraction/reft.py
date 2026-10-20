"""ReFT-family objects: LoReFT (arXiv:2404.03592) and AxBench ReFT-r1.

LoReFTObject — the low-rank affine subspace edit
    h  <-  h + scale * R^T (W h + b - R h)
with R [r, d] kept ORTHONORMAL by torch's orthogonal parametrization and
(W, b) a learned affine map into the subspace. This is NOT expressible as
a SteeringHandle addition (the edit depends linearly on h through two
learned maps), so LoReFT registers its OWN forward hook on the decoder
layer(s). It intervenes on PROMPT POSITIONS ONLY:

- generation: gated on prefill vs decode by seq_len — a seq_len > 1
  forward is the prefill (all positions are prompt positions -> edit); a
  seq_len == 1 forward is a KV-cached decode step (-> pass through
  untouched);
- training (full-sequence teacher-forced forwards, where seq_len > 1
  includes completion positions): the trainer sets per-sample prompt
  lengths via set_prompt_lens(), and only positions < prompt_len are
  edited.

Init: W is initialized to (a copy of) R and b to 0, so W h + b - R h = 0
and the untrained object is exactly the unsteered model — the harness's
zero-init convention. Defaults r=4, one layer (Phase-0 note:
docs/method_specs_s2.md was absent when this was written; rank and the
prefill/decode gating follow the plan text; position selection defaults
to ALL prompt positions).

ReFTr1Object — AxBench's LsReFT / "ReFT-r1" (verified against
stanfordnlp/axbench models/lsreft.py + interventions.py and paper
2501.17148; an earlier two-vector encoder/decoder variant here was NOT
their method and was rewritten):
- ONE vector w [d] plays encoder and steering direction alike. Their
  Linear has a bias, but it never enters forward, so none is kept here.
  Init w.fill_(0.01), their init — so the UNTRAINED object is
  deliberately NOT a no-op at scale 1 (unlike the harness's usual
  zero-init convention; scale=0 through apply() is still bit-exact
  unsteered, which is what the registry invariant tests exercise).
- Training forward at the trained layer, over the intervened positions
  (all non-BOS prompt+completion tokens, pads masked):
  z_i = ReLU(h_i . w); take the top-k of z across the sequence,
  m = mean of those k values (their CODE, not the paper's 1/|h|), and
  add m*w UNIFORMLY at every intervened position (not per-position).
- Loss = LM cross-entropy on the completions + lambda(step) * L1, where
  the L1 is the masked mean of the NON-top-k latents (penalize what the
  top-k dropped, never the survivors) and
  lambda(step) = l1_lambda * step / num_steps.
- Optimizer: AdamW lr 5e-3, weight_decay 0, linear LR decay to 0,
  grad-norm clip 1.0; w is held on the unit sphere every step (parallel
  gradient component projected out, then renormalized).
- After training, max_act is computed by a DETECTION pass over the
  training examples (max over examples of the max over non-BOS
  positions of ReLU(h . w), unsteered forward) and persisted with the
  artifact: AxBench steers at factor * max_act (this repo's generic scale
  sweeps keep the raw scale*w semantics of apply()).

DELIBERATE ADAPTATION — training data (read this before quoting parity):
AxBench trains LsReFT on the concept500 POSITIVE + NEGATIVE splits (72
curated examples each). In this SI-distillation harness the trainer
receives DistillExamples built by data/pairs.pairs_to_distill_examples,
which carry ONLY the positive side (the teacher's SI-prompted
completions on the fit prompts); Pair.neg (the no-SI completions — for
concept SIs generate_pairs falls to neg_mode="none", the closest analog
of AxBench's "unrelated" negatives) is DROPPED before this trainer ever
sees it. So every reft_r1 cell trains POSITIVES-ONLY. Wiring the negatives in would change the training
contract in the driver's pair plumbing (scripts/stage2_run.py /
data/pairs.py), so it is documented loudly here instead of half-done:
do NOT describe this arm as bit-faithful LsReFT training; the mechanism
(forward, loss shape, optimizer, calibration) is theirs, the data diet
is positives-only.

LOREFT PROBE-DRIFT POST-MORTEM (2026-08-24, safety_refusal wave objects):
train_loreft shipped a 5.2x-floor object (probe KL 0.4202 vs no-op floor
0.0803) while its OWN training CE improved monotonically the whole run
(per-epoch means 0.2177 → 0.1859 over 6 epochs, max stable ~0.95 — no
blowup anywhere in the 15,498 steps). The failure is a gradual
dissociation: 6 fixed-LR epochs keep compressing CE on the teacher's
SAMPLED tokens while the student's full DISTRIBUTION drifts away from
the SI-prompted teacher — which is what the driver's floor guard
measures (soft KL, lockstep.arm_probe_losses), and what CE cannot see.
Guards, mirroring lofit.py's (all defaults):

1. ``lr_decay`` / ``max_grad_norm``: linear LR decay to 0 and grad-norm
   clip 1.0 — the HF-Trainer defaults the reference (pyreft's trainer)
   runs under, which this port dropped (train_reft_r1 below always had
   both, per AxBench's config, and never tripped the guard).
2. The affine (W, b) group now trains at weight_decay=0.0 — torch AdamW's
   default 0.01 was a deviation from the reference (HF TrainingArguments
   default wd 0.0), and for LoReFT it is actively harmful: the no-op
   point is W=R (NOT W=0), so decoupled decay pushes the edit AWAY from
   identity every step — an anti-identity drift of exactly the kind the
   probe caught (over 15k steps the pull alone is a ~0.5x multiplicative
   shrink of W against R). The rotation group keeps wd=0 as before.
3. Per-epoch held-out early stopping on the guard's own probe metric with
   best-epoch restore, self-carving a probe slice when none is passed —
   lofit.py's helpers and semantics exactly (its module docstring has the
   full rationale; tiny pools skip the carve and keep the historical
   fixed-budget path).
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrizations

from ..hooks import PositionMaskedFit, SteeringHandle, steering
from ..models import get_decoder_layers, hidden_size
from . import ddp as _ddp
from .distill import (DistillExample, EarlyStopper, EarlyStopSpec,
                      _encode_aligned, _gather_completion_logits,
                      loss_gather_masks, steer_prompt_boundaries)
# The trainer-side held-out probe guard lives in lofit.py (same fix wave,
# same blast radius — see its module docstring); no cycle: lofit does not
# import reft.
from .lofit import HELDOUT_FRAC, carve_heldout, heldout_probe
from .traindata import Batch, collate


class LoReFTObject(PositionMaskedFit):
    """Orthonormal-subspace affine edit with prompt-position-only hooks.

    Position-masked (decode-only) fits: the edit is a per-position
    residual-stream function of h at the decoder-layer output, so a
    position mask is EXACT — but note what the leg means here. LoReFT's
    own identity is PROMPT-position intervention (prefill edit, decode
    pass-through, the module docstring's semantics), which is what
    ``fit_positions == "all"`` (the default) preserves bit-for-bit. Under
    ``set_fit_positions("response", H)`` that policy is DELIBERATELY
    inverted — the decode-only comparison leg is defined as "steer only
    the generated tokens" for every arm — so the hook edits the handle's
    masked positions instead: boundary-onward in a training forward
    (per-row boundaries via the yielded handle), the trailing H header
    positions of a generation prefill, and EVERY KV-cached decode step
    (the S==1 pass-through does not apply). Because this inverts the
    method's intrinsic position policy, the mode IS persisted in the
    artifact ("positions"), unlike the apply-everywhere arms: re-hydrating
    a response-fit LoReFT as prompt-only would silently run the wrong
    method (and no-op entirely under generated-only serving)."""

    #: Serve-side refusal of the decode-only leg.
    #: DISTINCT from FIT_MASK_REFUSAL, which says an object cannot express a
    #: mask; LoReFT can, and its masked fit stays supported and tested. The
    #: point is that it SHOULD NOT be served decode-only: LoReFT is
    #: prompt-positions-only BY IDENTITY (decode steps pass through
    #: untouched -- that is the method, not a default). For every other arm
    #: the decode-only leg RESTRICTS an all-position method, so the cell asks
    #: "does it survive restriction?". For LoReFT it is the COMPLEMENT: the
    #: intervention moves onto exactly the positions the method is defined to
    #: avoid, which runs a different method under this one's name. So it is
    #: excluded from class_c_arms AND from cross-phase serving, not merely
    #: left unlisted. Same shape as antidote's phase-invariance: an arm can
    #: be mechanically capable of a cell that means nothing.
    SERVE_MASK_REFUSAL = (
        "LoReFT intervenes on PROMPT positions only -- that is the method's "
        "identity, not a default -- so a decode-only serve does not restrict "
        "it, it relocates the edit onto the positions the method is defined "
        "to avoid, which is a different method under LoReFT's name")


    kind = "loreft"

    def __init__(self, model, layers: list[int], rank: int = 4, seed: int = 0):
        d = hidden_size(model)
        self.layers = list(layers)
        self.rank = rank
        self.rotate: dict[int, nn.Linear] = {}
        self.learned: dict[int, nn.Linear] = {}
        torch.manual_seed(seed)
        for l in self.layers:
            rot = nn.Linear(d, rank, bias=False)
            parametrizations.orthogonal(rot)  # rows of weight stay orthonormal
            lin = nn.Linear(d, rank, bias=True)
            with torch.no_grad():
                lin.weight.copy_(rot.weight)  # W=R, b=0 -> identity at init
                lin.bias.zero_()
            self.rotate[l] = rot
            self.learned[l] = lin
        self._decoder = get_decoder_layers(model)
        self._prompt_lens: torch.Tensor | None = None

    @property
    def trainable(self) -> list[nn.Parameter]:
        """All trainable params. CAUTION: never put the rotate params under
        an optimizer with weight decay — decoupled decay on the householder
        `original` collapses the materialized R to zero (see train_loreft,
        which uses a wd=0 group for them)."""
        params: list[nn.Parameter] = []
        for l in self.layers:
            params += list(self.rotate[l].parameters())
            params += list(self.learned[l].parameters())
        return params

    def n_params(self) -> int:
        return sum(p.numel() for p in self.trainable)

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def set_prompt_lens(self, lens: torch.Tensor | None) -> None:
        """Per-sample prompt lengths for full-sequence (training) forwards;
        None restores generation-time prefill/decode gating. Prompt-mode
        machinery only: a response-mode object derives its mask from the
        fit handle (set per-row boundaries on the handle apply() yields),
        so mixing the two would silently double-specify the positions."""
        if lens is not None and self.fit_positions != "all":
            raise RuntimeError(
                "set_prompt_lens is LoReFT's prompt-mode machinery; this "
                f"object is in fit_positions={self.fit_positions!r} mode — "
                "set per-row boundaries on the mask handle apply() yields "
                "(handle.set_prompt_lens(steer_prompt_boundaries(batch, "
                "H))) instead")
        self._prompt_lens = lens

    def R(self, layer: int) -> torch.Tensor:
        """The materialized orthonormal subspace matrix [r, d]."""
        return self.rotate[layer].weight

    def _make_hook(self, layer: int, scale: float,
                    mask_handle: SteeringHandle | None = None):
        rot, lin = self.rotate[layer], self.learned[layer]

        def hook(_mod, _args, output):
            if scale == 0.0:  # exact no-op
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            B, S, _ = hidden.shape
            if mask_handle is not None:
                # Response-mode fit: the handle's mask is the ONE position
                # source — decode steps (S == 1, past the boundary) are
                # edited, a prefill edits its trailing header positions,
                # and a training forward follows the per-row boundaries
                # set on the handle. The prompt-mode gating below (decode
                # pass-through + positions < prompt_len) does not apply.
                h32 = hidden.float()
                delta = (lin(h32) - rot(h32)) @ rot.weight  # [B, S, d]
                steered = hidden + (scale * delta).to(hidden.dtype)
                steered = self._fit_masked(mask_handle, steered, hidden)
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered
            if self._prompt_lens is None and S == 1:
                return output  # KV-cached decode step: pass through
            h32 = hidden.float()
            delta = (lin(h32) - rot(h32)) @ rot.weight  # [B, S, d]
            if self._prompt_lens is not None:
                local = torch.arange(S, device=hidden.device)[None, :]
                mask = local < self._prompt_lens.to(hidden.device)[:, None]
                delta = delta * mask.unsqueeze(-1)
            steered = hidden + (scale * delta).to(hidden.dtype)
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        return hook

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mask_handle = self._fit_mask_handle(model)
        handles = [
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

    def to_(self, device) -> None:
        for l in self.layers:
            self.rotate[l].to(device)
            self.learned[l].to(device)

    def save(self, path: Path) -> None:
        # "positions" persists the object's INTRINSIC position policy —
        # "prompt" (the method's own identity, the default fit) or
        # "response" (a decode-only-fit artifact, which must re-hydrate
        # as such or it silently runs the wrong method). The
        # apply-everywhere arms deliberately do NOT persist a fit mask;
        # LoReFT does because positions are part of what the object IS.
        torch.save(
            {"kind": self.kind,
             "layers": self.layers,
             "rank": self.rank,
             "positions": ("response" if self.fit_positions == "response"
                           else "prompt"),
             "response_header_len": self.fit_response_header_len,
             "rotate": {str(l): self.rotate[l].state_dict()
                        for l in self.layers},
             "learned": {str(l): self.learned[l].state_dict()
                         for l in self.layers}},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "LoReFTObject":
        layers = [int(l) for l in blob["layers"]]
        obj = cls(model, layers, rank=int(blob["rank"]))
        for l in layers:
            obj.rotate[l].load_state_dict(blob["rotate"][str(l)])
            obj.learned[l].load_state_dict(blob["learned"][str(l)])
        if blob.get("positions", "prompt") == "response":
            obj.set_fit_positions("response",
                                  blob.get("response_header_len"))
        return obj


def train_loreft(
    model,
    tok,
    obj: LoReFTObject,
    examples: list[DistillExample],
    *,
    lr: float = 5e-3,
    epochs: int = 6,
    batch_size: int = 2,
    max_len: int = 768,
    seed: int = 0,
    log_fn=None,
    positions: str = "all",
    response_header_len: int | None = None,
    lr_decay: bool = True,
    max_grad_norm: float | None = 1.0,
    heldout: list[DistillExample] | None = None,
    heldout_frac: float = HELDOUT_FRAC,
    early_stop: EarlyStopSpec | None = None,
    diagnostics: dict | None = None,
) -> list[float]:
    """LoReFT training: cross-entropy of the edited no-SI student on the
    teacher completions (the paper's LM objective), edits applied at
    prompt positions only via set_prompt_lens.

    Probe-drift guards (module docstring's post-mortem — all defaults):
    ``lr_decay`` (linear to 0) and ``max_grad_norm`` (clip; None disables)
    restore the reference trainer's HF defaults. ``heldout`` +
    ``early_stop`` add per-epoch early stopping on the floor guard's own
    soft-KL-to-teacher probe with best-epoch restore; when ``heldout`` is
    None a probe slice is self-carved from ``examples`` (``heldout_frac``;
    pools too small to carve skip the stop entirely — the historical
    fixed-budget path). ``diagnostics`` (optional dict) is filled with the
    probe curve and stopping record — LoReFTObject has no metadata channel,
    so this and ``log_fn`` are how the record leaves the trainer.

    ``positions="response"`` selects the decode-only fit leg instead: the
    edit applies from each row's boundary on (header-steered when
    ``response_header_len`` is set; H is derived ONCE by the caller, never
    here) via the mask handle apply() yields, and the CE gathers exactly
    the completion positions that mask covers (``loss_gather_masks`` — the
    mask→loss wiring). This deliberately inverts LoReFT's prompt-only
    identity; see the class docstring. The default is byte-identical to
    the historical prompt-only fit."""
    if positions != "all":
        obj.set_fit_positions(positions, response_header_len)
    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    obj.to_(device)
    # Held-out probe slice for the per-epoch stop: an explicit ``heldout``
    # wins; otherwise self-carve (no-op on pools too small to probe).
    if heldout is None:
        examples, heldout = carve_heldout(examples, heldout_frac, seed)
    # The orthogonal (householder) parametrization is DESTROYED by decoupled
    # weight decay on its unconstrained `original` parameter: AdamW's default
    # wd=0.01 collapses the materialized R to exactly zero within a few
    # steps (verified on torch 2.x). The rotation group therefore trains
    # with weight_decay=0. The affine (W, b) group is ALSO wd=0 now (module
    # docstring, guard 2): its no-op point is W=R, not W=0, so decay pushes
    # the edit AWAY from identity — and the reference trainer's HF default
    # is wd 0.0 anyway; the earlier torch-default 0.01 was a deviation.
    rotate_params: list[nn.Parameter] = []
    affine_params: list[nn.Parameter] = []
    for l in obj.layers:
        rotate_params += list(obj.rotate[l].parameters())
        affine_params += list(obj.learned[l].parameters())
    opt = torch.optim.AdamW(
        [{"params": rotate_params, "weight_decay": 0.0},
         {"params": affine_params, "weight_decay": 0.0}],
        lr=lr,
    )
    sched = None
    encoded = [_encode_aligned(tok, ex, max_len)[1] for ex in examples]
    batches: list[Batch] = [
        collate(encoded[i : i + batch_size], tok.pad_token_id)
        for i in range(0, len(encoded), batch_size)
    ]
    if lr_decay:
        total_steps = max(epochs * len(batches), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: max(0.0, 1.0 - s / total_steps))

    # Row-sharded fit context (extraction.ddp): under the ambient context
    # each rank forwards only its row slice of every batch — same batches,
    # same order, same effective batch and step count as single-process
    # (the ddp module's exact-semantics contract). Byte-identical single-
    # process path when no context is set.
    par = _ddp.active()
    sharded = par is not None and par.enabled
    all_params = rotate_params + affine_params

    history: list[float] = []
    step = 0
    stopper = EarlyStopper(early_stop or EarlyStopSpec()) if heldout else None
    heldout_curve: list[float] = []
    probe_floor: float | None = None
    stopped_epoch: int | None = None
    epochs_run = 0

    def _probe(epoch: int) -> float:
        nonlocal probe_floor
        act, floor = heldout_probe(model, tok, obj, heldout, max_len=max_len)
        probe_floor = floor
        heldout_curve.append(act)
        if log_fn:
            log_fn({"loreft/heldout_probe": act,
                    "loreft/heldout_probe_floor": floor,
                    "epoch": epoch}, step)
        return act

    if stopper is not None:
        # Pre-training baseline point (curve only — never a stopper update,
        # so the init can never become the restore target; lockstep's rule).
        _probe(-1)
    for epoch in range(epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        for bi in order:
            b_full = batches[bi]
            rows = _ddp.rows_for_rank(b_full.input_ids.shape[0], par)
            step_loss = None
            k_local = 0
            if rows:
                student_b = (_ddp.slice_batch(b_full, rows) if sharded
                             else b_full).to(device)
                s_gather = None
                if obj.fit_positions == "all":
                    try:
                        obj.set_prompt_lens(student_b.prompt_lens)
                        with obj.apply(model, scale=1.0):
                            logits = model(
                                input_ids=student_b.input_ids,
                                attention_mask=student_b.attention_mask,
                            ).logits
                    finally:
                        obj.set_prompt_lens(None)
                else:
                    # Decode-only leg: per-row boundaries on the yielded
                    # mask handle BEFORE the forward, CE gathered from the
                    # SAME mask.
                    with obj.apply(model, scale=1.0) as mask_handle:
                        mask_handle.set_prompt_lens(steer_prompt_boundaries(
                            student_b, mask_handle.response_header_len))
                        logits = model(
                            input_ids=student_b.input_ids,
                            attention_mask=student_b.attention_mask,
                        ).logits
                        s_gather, _t_gather, _keep = loss_gather_masks(
                            mask_handle, student_b)
                s_logits = _gather_completion_logits(logits, student_b,
                                                     gather=s_gather)
                labels = student_b.labels[:, 1:]
                targets = (labels[s_gather] if s_gather is not None
                           else labels[labels != -100])
                step_loss = F.cross_entropy(s_logits.float(), targets)
                k_local = int(targets.numel())
            if sharded:
                # CE is a mean over target positions: scale this rank's
                # share by k/K so the summed gradients equal the single-
                # process step exactly (up to fp reduction order).
                (k_glob,) = _ddp.allreduce_counts([k_local], par)
                opt.zero_grad()
                if step_loss is not None:
                    scaled = _ddp.scale_loss_parts(
                        [(step_loss, k_local, k_glob)], device)
                    scaled.backward()
                    share = float(scaled.detach())
                else:
                    share = 0.0
                (loss_val,) = _ddp.allreduce_grads_and_stats(
                    all_params, [share], par)
            else:
                opt.zero_grad()
                step_loss.backward()
                loss_val = float(step_loss.detach())
            # Clip AFTER the all-reduce (sharded): the global gradient
            # every rank now holds is the tensor the single-process clip
            # sees (clipping local partials before the sum would not be).
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(obj.trainable, max_grad_norm)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            history.append(loss_val)
            if log_fn:
                log_fn({"loreft/ce_loss": loss_val, "epoch": epoch}, step)
        epochs_run = epoch + 1
        if stopper is not None:
            act = _probe(epoch)
            if stopper.update(act, obj.trainable):
                stopper.restore_best(obj.trainable)
                stopped_epoch = epoch + 1
                if log_fn:
                    log_fn({"loreft/stopped_epoch": stopped_epoch,
                            "loreft/best_epoch": stopper.best_epoch}, step)
                break
    if stopper is not None and stopped_epoch is None:
        # Budget cap reached with a better epoch inside the patience
        # window: ship the best epoch, not the last (lockstep's rule).
        stopper.restore_best(obj.trainable)
    if sharded:
        # After the restore, so the assertion rules on the params that SHIP
        # (a rank-divergent probe decision would surface here, loudly).
        _ddp.assert_replicas_equal("loreft", all_params, par)
    if diagnostics is not None:
        diagnostics.update({
            "enabled": stopper is not None,
            "metric": "soft-KL-to-teacher probe (lockstep.arm_probe_losses) "
                      "on held-out rows — the floor guard's own metric",
            "n_heldout": len(heldout),
            # curve[0] is the pre-training baseline (epoch -1), then one
            # point per completed epoch.
            "heldout_curve": heldout_curve,
            "noop_floor_probe": probe_floor,
            "best_epoch": stopper.best_epoch if stopper else None,
            "best_heldout_probe": (
                stopper.best if stopper and math.isfinite(stopper.best)
                else None),
            "stopped_epoch": stopped_epoch,
            "epochs_run": epochs_run,
        })
    return history


def _intervened_mask(
    B: int, S: int, attn_mask: torch.Tensor | None, device
) -> torch.Tensor:
    """[B, S] bool mask of ReFT-r1's intervened positions: every real
    (non-pad) token EXCEPT each row's first one (the BOS). argmax over
    the 2D attention mask finds the first real position under right and
    left padding alike; without a mask (unbatched/unpadded forwards) the
    BOS is index 0 exactly."""
    if attn_mask is not None:
        am = attn_mask.to(device)
        valid = am.bool()
        bos = am.argmax(dim=1)
    else:
        valid = torch.ones(B, S, dtype=torch.bool, device=device)
        bos = torch.zeros(B, dtype=torch.long, device=device)
    pos = torch.arange(S, device=device)[None, :]
    return valid & (pos != bos[:, None])


class ReFTr1Object:
    """AxBench LsReFT: one vector, top-k-mean uniform addition.

    See the module docstring for the verified method spec and the
    positives-only training-data adaptation."""

    kind = "reft_r1"

    def __init__(self, model, layer: int, topk: int = 8, seed: int = 0):
        # `seed` is kept for constructor-signature stability (configs pass
        # it); the AxBench init is the deterministic 0.01 fill, so it is
        # unused here.
        d = hidden_size(model)
        self.layer = layer
        self.topk = topk
        # Their init: every coordinate 0.01. Deliberately NOT a no-op when
        # untrained and applied at scale != 0 (module docstring); training
        # puts w on the unit sphere from the first step onward.
        self.w = nn.Parameter(torch.full((d,), 0.01))
        # Per-concept/instruction steering calibration, filled by
        # train_reft_r1's detection pass and persisted with the artifact
        # (AxBench steers at factor * max_act; eval never retrains).
        self.max_act: float | None = None
        self._last_nontopk: torch.Tensor | None = None
        self._last_valid: torch.Tensor | None = None

    @property
    def trainable(self) -> list[nn.Parameter]:
        return [self.w]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.trainable)

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def apply(self, model, scale: float = 1.0):
        """Eval-time steering: plain ADDITION of w at the trained layer,
        all positions, one scale — i.e. scale*w. The core suites' generic
        scale sweeps use this raw semantics; AxBench's own protocol
        multiplies its factor by the persisted max_act first."""
        return steering(model, {self.layer: self.w}, scale=scale)

    @contextmanager
    def latent_context(self, model, attn_mask: torch.Tensor | None = None):
        """Training-time forward path (their LsReFT forward): at the
        trained layer, z = ReLU(h . w) over the intervened positions,
        m = mean of the top-k z values per row, and h <- h + m*w applied
        UNIFORMLY at every intervened position. Stashes the non-top-k
        latents (self._last_nontopk) and the intervened-position mask
        (self._last_valid) for the trainer's L1 term."""
        decoder = get_decoder_layers(model)

        def hook(_mod, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h32 = hidden.float()
            w = self.w.to(h32.device)
            z = F.relu(h32 @ w)  # [B, S]
            B, S = z.shape
            valid = _intervened_mask(B, S, attn_mask, z.device)
            z = z * valid.to(z.dtype)
            k = min(self.topk, S)
            vals, idx = z.topk(k, dim=1)  # [B, k]
            z_nontopk = z.scatter(1, idx, torch.zeros_like(vals))
            m = vals.mean(dim=1)  # [B] — mean over the k values (their code)
            self._last_nontopk = z_nontopk
            self._last_valid = valid
            add = (m[:, None] * valid.to(h32.dtype))[:, :, None] * w
            steered = hidden + add.to(hidden.dtype)
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        h = decoder[self.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()
            self._last_nontopk = None
            self._last_valid = None

    def renormalize_(self) -> None:
        """Project w back to unit norm (called after EVERY optimizer
        step — the AxBench invariant)."""
        self.w.data = self.w.data / self.w.data.norm().clamp_min(1e-8)

    def to_(self, device) -> None:
        for p in self.trainable:
            p.data = p.data.to(device)

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind, "layer": self.layer, "topk": self.topk,
             "w": self.w.detach().float().cpu(),
             "max_act": self.max_act},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "ReFTr1Object":
        if "w" not in blob:
            raise ValueError(
                "reft_r1 artifact predates the one-vector LsReFT rewrite "
                "(has no 'w') — retrain it; the old two-vector object was "
                "not AxBench's method"
            )
        obj = cls(model, layer=int(blob["layer"]), topk=int(blob["topk"]))
        obj.w.data = blob["w"].clone()
        max_act = blob.get("max_act")
        obj.max_act = float(max_act) if max_act is not None else None
        return obj


@torch.no_grad()
def compute_reft_r1_max_act(model, obj: ReFTr1Object, batches: list[Batch],
                            device=None) -> float:
    """AxBench's steering calibration: a DETECTION pass (unsteered
    forwards) over the concept's examples — max over examples of the max
    over intervened (non-BOS, non-pad) positions of ReLU(h . w)."""
    from .capture import capture_layer_outputs

    if device is None:
        device = next(model.parameters()).device
    w = obj.w.detach().float()
    best = 0.0
    for b in batches:
        b = b.to(device)
        with capture_layer_outputs(model, [obj.layer]) as cap:
            model(input_ids=b.input_ids, attention_mask=b.attention_mask)
        h = cap[obj.layer].float()
        z = F.relu(h @ w.to(h.device))
        B, S = z.shape
        valid = _intervened_mask(B, S, b.attention_mask, z.device)
        z = z * valid.to(z.dtype)
        best = max(best, float(z.max()))
    return best


def train_reft_r1(
    model,
    tok,
    obj: ReFTr1Object,
    examples: list[DistillExample],
    *,
    lr: float = 5e-3,
    epochs: int = 6,
    batch_size: int = 2,
    max_len: int = 768,
    l1_lambda: float = 0.005,
    seed: int = 0,
    log_fn=None,
) -> list[float]:
    """LsReFT training (their config: AdamW lr 5e-3 wd 0, linear LR decay
    to 0, grad clip 1.0): LM cross-entropy on the completions with the
    top-k-mean uniform addition active, plus lambda(step) * L1 where the
    L1 is the masked mean of the NON-top-k latents (penalizing the mass
    the top-k dropped — the sparsity pressure; the earlier code penalized
    the survivors, which is sign-flipped) and
    lambda(step) = l1_lambda * step / num_steps. w is held on the unit
    sphere every step. Ends with the max_act detection pass (module
    docstring); training data is positives-only (deliberate adaptation,
    module docstring)."""
    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    obj.to_(device)
    opt = torch.optim.AdamW(obj.trainable, lr=lr, weight_decay=0.0)

    encoded = [_encode_aligned(tok, ex, max_len)[1] for ex in examples]
    batches: list[Batch] = [
        collate(encoded[i : i + batch_size], tok.pad_token_id)
        for i in range(0, len(encoded), batch_size)
    ]
    total_steps = max(epochs * len(batches), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: max(0.0, 1.0 - s / total_steps)  # linear decay to 0
    )

    history: list[float] = []
    step = 0
    for epoch in range(epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        for bi in order:
            student_b = batches[bi].to(device)
            with obj.latent_context(
                model, attn_mask=student_b.attention_mask
            ):
                logits = model(
                    input_ids=student_b.input_ids,
                    attention_mask=student_b.attention_mask,
                ).logits
                z_nontopk = obj._last_nontopk
                valid = obj._last_valid
            s_logits = _gather_completion_logits(logits, student_b)
            labels = student_b.labels[:, 1:]
            targets = labels[labels != -100]
            ce = F.cross_entropy(s_logits.float(), targets)
            lam = l1_lambda * (step / total_steps)
            nv = valid.to(z_nontopk.dtype)
            l1 = (z_nontopk * nv).sum() / nv.sum().clamp_min(1.0)
            step_loss = ce + lam * l1
            opt.zero_grad()
            step_loss.backward()
            with torch.no_grad():
                # Keep the update tangent to the unit sphere: project the
                # component of the gradient parallel to w out, then step,
                # then renormalize (either half alone drifts under Adam's
                # per-coordinate scaling).
                if obj.w.grad is not None:
                    unit = obj.w / obj.w.norm().clamp_min(1e-8)
                    obj.w.grad -= (obj.w.grad @ unit) * unit
            torch.nn.utils.clip_grad_norm_(obj.trainable, 1.0)
            opt.step()
            obj.renormalize_()
            sched.step()
            step += 1
            loss_val = float(step_loss.detach())
            history.append(loss_val)
            if log_fn:
                log_fn({"reft_r1/loss": loss_val,
                        "reft_r1/ce": float(ce), "reft_r1/l1": float(l1),
                        "reft_r1/lambda": lam, "epoch": epoch}, step)
    # AxBench steering calibration, persisted with the artifact so eval
    # never retrains (save() writes it).
    obj.max_act = compute_reft_r1_max_act(model, obj, batches, device=device)
    return history
