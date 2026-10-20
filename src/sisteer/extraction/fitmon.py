"""Production fit monitoring: dual-KL ratio probes, gradient-norm taps,
per-arm norm groups, probe cadence, and the non-finite abort.

Lifted from the lora_r1_all HPO harness (scripts/tune_lora_all.py
@ 8ded0ea) into the library so BOTH consumers — the generalized tuning
harness (scripts/tune_arm.py) and the production fit driver
(scripts/train_two_si_objects.py) — import ONE implementation instead
of each carrying a copy.

The two KL ratios (spec Fixed 9 — both are logged per cell, because a
dead arm and a wrong-direction arm separate only when both are read
together):

  residual      KL(prompted || steered) / KL(prompted || unsteered)
                (the lane's floor_ratio: 0 = reproduces prompting,
                1 = no better than doing nothing)
  displacement  KL(steered || unsteered) / KL(prompted || unsteered)
                (how far the object has moved the model, in units of how
                far prompting moves it)

The denominator KL(prompted || unsteered) — the no-op floor on the probe
examples — is computed ONCE per probe instance and cached (the base is
frozen, so it cannot move).

Everything here is OBSERVATIONAL: no RNG draws, no parameter writes, and
tensor hooks that return None — so attaching the monitors cannot perturb
a training trajectory (the lockstep bitwise contract holds monitors-on;
pinned by tests/test_lockstep.py).

SYNC DISCIPLINE: the monitors must never ADD per-step CPU<->GPU
synchronisations — a float()/.item() per metric per step forces a
cudaStreamSynchronize each, and observability that tanks the duty cycle
defeats its own purpose. Per-step scalars (grad norms, object norms) are
therefore accumulated as 0-dim GPU TENSORS in a buffer and .item()'d ONCE
per logging interval (:class:`ProbeCadence.log_every`); the KL probe is
cadence-gated (dense -> sparse) and syncs only when it runs. Consequence:
the non-finite abort fires at the next FLUSH, i.e. at most ``log_every``
steps after the divergence — NaNs only propagate in between, so nothing
recoverable is lost.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .distill import (
    _encode_aligned,
    _gather_completion_logits,
    per_example_means,
    soft_divergence_per_token,
    steer_prompt_boundaries,
)
from .subsample import sha_hex, sha_rank_indices
from .traindata import collate


class TrainingDiverged(RuntimeError):
    """ANY non-finite required metric — step loss, grad norms, object
    norms, probe ratios — aborts the arm and records it failed; a
    diverged fit is never shipped."""


#: The tuning harness's historical name for the same abort (tune_lora_all's
#: ``TrialDiverged``) — one exception class, two vocabularies.
TrialDiverged = TrainingDiverged


def assert_finite(record: dict, *, step=None, context: str = "") -> None:
    """Raise :class:`TrainingDiverged` if any numeric value in ``record``
    is non-finite, naming the offending key(s). The production guard is
    deliberately over the FULL metric set, not just the step loss — a NaN
    grad norm or probe ratio is the same divergence one step earlier."""
    bad = {k: v for k, v in record.items()
           if isinstance(v, (int, float)) and not isinstance(v, bool)
           and not math.isfinite(v)}
    if bad:
        where = f" at step {step}" if step is not None else ""
        who = f" [{context}]" if context else ""
        raise TrainingDiverged(
            f"non-finite metric(s){who}{where}: "
            + ", ".join(f"{k}={v}" for k, v in sorted(bad.items())))


# --------------------------------------------------------------------------
# Step-level divergence guard (fit-wave hardening, 2026-08-27): per-epoch
# early stopping rules once per epoch, so a doomed trial on a slow model
# (qwen psr ~8h/epoch) burned ~15h before its epoch-boundary stop even
# though its loss was visibly exploding step-to-step (a T=4 trial's own-T
# heldout went 0.002 -> 0.772, 386x, inside epoch 1). This guard reads the
# per-step training loss the trainers ALREADY hold as a python float (zero
# new CPU<->GPU syncs) and raises TrainingDiverged in STEPS, not epochs.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StepDivergenceSpec:
    """When a run counts as diverging, step-to-step. Tuned so a genuinely
    exploding run dies in minutes while a healthy run's transient bf16
    spikes and bs=1 step-loss noise (~10x step-to-step on the live canary)
    survive:

    - the raw step loss is smoothed by a MEDIAN over the last ``smooth``
      optimizer steps, so no single-step spike can contribute a verdict;
    - the baseline is the running MINIMUM of that smoothed series — a
      divergence cannot drag its own baseline up, and a descending or
      flat-noisy run never sits above its own best for long;
    - clause A (sustained explosion): smoothed loss >= ``factor`` x best
      for ``patience`` CONSECUTIVE optimizer steps (one dip resets the
      counter — oscillating plateaus cannot accumulate a verdict);
    - clause B (fast monotone blow-up): the smoothed loss rose STRICTLY
      monotonically across the last ``rise_window`` steps by a cumulative
      ``rise_factor`` — catches a from-the-start explosion whose own
      first values are the baseline, faster than clause A's patience;
    - a non-finite loss is an immediate verdict (no smoothing, no
      patience — NaN only propagates);
    - no clause rules before ``warmup_steps`` optimizer steps (and never
      before the median window fills), so the first noisy steps of a
      healthy fit cannot self-condemn.

    A verdict raises :class:`TrainingDiverged` — the established abort
    vocabulary: HPO trials self-publish a terminal summary (status
    ``nan_diverged``) and exit cleanly; the production lockstep loop
    marks THAT ARM diverged (recorded failed, no artifact) while its
    group partners train on."""

    factor: float = 10.0
    patience: int = 20
    smooth: int = 5
    rise_window: int = 8
    rise_factor: float = 50.0
    warmup_steps: int = 10

    def __post_init__(self):
        for name in ("patience", "smooth", "rise_window"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.factor <= 1.0 or self.rise_factor <= 1.0:
            raise ValueError("factor/rise_factor must be > 1")


def resolve_divergence_guard(flag, *, tag: str):
    """The trainers' one flag resolver: ``None``/``True`` -> a guard at
    the production defaults (default ON), ``False`` -> off (``None``), a
    :class:`StepDivergenceSpec` -> a guard at that tuning."""
    if flag is False:
        return None
    if flag is None or flag is True:
        return StepDivergenceGuard(StepDivergenceSpec(), tag=tag)
    if isinstance(flag, StepDivergenceSpec):
        return StepDivergenceGuard(flag, tag=tag)
    raise ValueError(
        f"divergence_guard must be True/False/None or a StepDivergenceSpec, "
        f"got {flag!r}")


class StepDivergenceGuard:
    """Feed :meth:`step` once per optimizer step with the loss the trainer
    already synced for its history; it raises :class:`TrainingDiverged`
    (message prefixed ``step-divergence:``) when the spec's clauses rule.
    Pure python floats — observational, no RNG, no writes, no syncs."""

    def __init__(self, spec: StepDivergenceSpec, *, tag: str = ""):
        self.spec = spec
        self.tag = tag
        self._raw: list[float] = []       # last `smooth` raw losses
        self._smoothed: list[float] = []  # last `rise_window` medians
        self._best: float | None = None
        self._over = 0                    # consecutive steps over factor*best
        self._seen = 0

    def step(self, step_index: int, loss: float) -> None:
        import statistics

        s = self.spec
        who = f" [{self.tag}]" if self.tag else ""
        if not math.isfinite(loss):
            raise TrainingDiverged(
                f"step-divergence{who}: non-finite training loss {loss} at "
                f"optimizer step {step_index}")
        self._seen += 1
        self._raw.append(float(loss))
        if len(self._raw) > int(s.smooth):
            self._raw.pop(0)
        if len(self._raw) < int(s.smooth):
            return  # median window not filled — no verdicts yet
        med = statistics.median(self._raw)
        self._best = med if self._best is None else min(self._best, med)
        self._smoothed.append(med)
        if len(self._smoothed) > int(s.rise_window):
            self._smoothed.pop(0)
        if self._seen < int(s.warmup_steps):
            return
        # Clause A: sustained explosion over the running-min baseline.
        bar = s.factor * max(self._best, 1e-12)
        self._over = self._over + 1 if med >= bar else 0
        if self._over >= int(s.patience):
            raise TrainingDiverged(
                f"step-divergence{who}: smoothed training loss has sat >= "
                f"{s.factor}x its best ({self._best:.6g}) for {s.patience} "
                f"consecutive optimizer steps — latest {med:.6g} at step "
                f"{step_index}; aborting in steps instead of waiting for "
                "the epoch-boundary early stop")
        # Clause B: fast strictly-monotone blow-up across the window.
        w = self._smoothed
        if (len(w) == int(s.rise_window)
                and all(b > a for a, b in zip(w, w[1:]))
                and w[-1] >= s.rise_factor * max(w[0], 1e-12)):
            raise TrainingDiverged(
                f"step-divergence{who}: smoothed training loss rose "
                f"monotonically {w[-1] / max(w[0], 1e-12):.1f}x over the "
                f"last {s.rise_window} optimizer steps ({w[0]:.6g} -> "
                f"{w[-1]:.6g}) at step {step_index}")


def pin_attention_backends() -> None:
    """Disable the cuDNN SDPA backend (port of the lora worktree fix,
    commit 8468792; the local ``gpu-util`` skill documents the footgun).

    cuDNN's SDPA backend re-plans the attention graph ON CPU for every new
    (batch, seqlen) shape; with ragged batches that measured ~1.9 s/step of
    pure CPU planning (r5_t00 profiler trace: 28.2s of
    _cudnn_attention_{forward,backward} CPU time in a 42.8s window, GPU
    kernel duty 33%). The flash/mem-efficient SDPA backends have no
    per-shape planning cost. One shared helper so the production driver
    and the tuning harness pin the same backends."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)


# --------------------------------------------------------------------------
# Probe cadence — dense early (instability hides from sparse evals), then
# periodic. Config-tunable via train_kwargs.default.{probe_every,
# dense_every, dense_until}.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeCadence:
    """When the (expensive) KL probe runs — every ``dense_every`` steps
    until ``dense_until``, then every ``probe_every`` steps (the tuning
    harness's schedule, now shared with production) — and when buffered
    per-step scalars are FLUSHED to the log (``log_every``; the sync
    discipline in the module docstring)."""

    probe_every: int = 250
    dense_every: int = 25
    dense_until: int = 250
    log_every: int = 25

    def __post_init__(self):
        for name in ("probe_every", "dense_every", "log_every"):
            v = getattr(self, name)
            if int(v) < 1:
                raise ValueError(f"{name} must be >= 1, got {v}")

    def due(self, step: int) -> bool:
        dense = step <= self.dense_until and step % self.dense_every == 0
        return dense or step % self.probe_every == 0

    def log_due(self, step: int) -> bool:
        return step % self.log_every == 0 or self.due(step)

    @classmethod
    def from_train_kwargs(cls, tk: dict) -> "ProbeCadence":
        """The config seam (train_kwargs.default.probe_every /
        dense_every / dense_until / log_every; absent keys keep the
        defaults)."""
        return cls(
            probe_every=int(tk.get("probe_every", cls.probe_every)),
            dense_every=int(tk.get("dense_every", cls.dense_every)),
            dense_until=int(tk.get("dense_until", cls.dense_until)),
            log_every=int(tk.get("log_every", cls.log_every)))


# --------------------------------------------------------------------------
# The two KL ratios, on a fixed held-out probe (port of tune_lora_all's
# KLRatioProbe, + the object-inject seam the soft-token arms need).
# --------------------------------------------------------------------------


class KLRatioProbe:
    """Fixed sha-rank-selected held-out probe measuring, per call:

      residual_kl      KL(prompted || steered)        (the floor numerator)
      displacement_kl  KL(steered || unsteered)       (the movement numerator)

    against a once-computed denominator KL(prompted || unsteered) — the
    no-op floor on THESE examples. All three are per-example completion-
    token means averaged over examples (the chunking-invariant estimator
    the lane's probes use), teacher rendered WITH the SI, student without,
    completion positions index-aligned by _encode_aligned's contract.

    Selection is by sha256 rank of the student rendering
    (:func:`subsample.sha_rank_indices` — never a pool prefix, which is
    single-family on the sorted pools), so probe subsets NEST: the n=64
    probe's examples are a subset of the n=256 probe's over the same pool.
    ``selected_sha8`` records which examples the estimate averaged.

    ``inject`` (optional): the object's prompt injection applied to the
    STUDENT rendering before encoding — the ``_encode_arm_batches``
    convention, needed for the soft-token arms whose intervention lives in
    the prompt ids (a sentinel absent from the ids makes their apply() a
    silent no-op). Teacher side is never injected. Default None keeps the
    tuning harness's byte-exact behaviour for the identity-inject arms."""

    def __init__(self, tok, examples, *, max_len: int, batch_size: int,
                 n: int = 64, inject=None):
        keys = [ex.prompt_without for ex in examples]
        idx = sha_rank_indices(keys, n)
        sel = [examples[i] for i in idx]
        self.selected_idx = list(idx)
        self.selected_sha8 = tuple(sha_hex(keys[i])[:8] for i in idx)
        if inject is not None:
            from .distill import DistillExample

            sel = [DistillExample(prompt_with_si=ex.prompt_with_si,
                                  prompt_without=inject(ex.prompt_without),
                                  completion=ex.completion,
                                  instruction_idx=ex.instruction_idx)
                   for ex in sel]
        enc = [_encode_aligned(tok, ex, max_len) for ex in sel]
        bs = max(min(int(batch_size), max(len(enc), 1)), 1)
        self.batches = []
        for i in range(0, len(enc), bs):
            chunk = enc[i:i + bs]
            self.batches.append(
                (collate([t for t, _ in chunk], tok.pad_token_id),
                 collate([s for _, s in chunk], tok.pad_token_id)))
        self.n = len(sel)
        self.denom: float | None = None

    @staticmethod
    def _counts(batch) -> list[int]:
        return [int(c) for c in (batch.labels[:, 1:] != -100).sum(-1)]

    @torch.no_grad()
    def measure(self, model, obj=None) -> dict:
        device = next(model.parameters()).device
        res_vals, disp_vals, floor_vals = [], [], []
        for teacher_b, student_b in self.batches:
            teacher_b = teacher_b.to(device)
            student_b = student_b.to(device)
            counts = self._counts(student_b)
            t_logits = _gather_completion_logits(
                model(input_ids=teacher_b.input_ids,
                      attention_mask=teacher_b.attention_mask).logits,
                teacher_b)
            u_logits = _gather_completion_logits(
                model(input_ids=student_b.input_ids,
                      attention_mask=student_b.attention_mask).logits,
                student_b)
            t_logp = torch.log_softmax(t_logits.float(), -1)
            u_logp = torch.log_softmax(u_logits.float(), -1)
            if self.denom is None:
                floor_vals.extend(per_example_means(
                    soft_divergence_per_token("soft", t_logp, u_logp),
                    counts))
            if obj is not None:
                with obj.apply(model, scale=1.0) as _ctx:
                    self._set_boundaries(_ctx, student_b)
                    s_logits = _gather_completion_logits(
                        model(input_ids=student_b.input_ids,
                              attention_mask=student_b.attention_mask).logits,
                        student_b)
                s_logp = torch.log_softmax(s_logits.float(), -1)
                # KL(prompted || steered): forward KL, target = teacher.
                res_vals.extend(per_example_means(
                    soft_divergence_per_token("soft", t_logp, s_logp),
                    counts))
                # KL(steered || unsteered): forward KL, target = steered.
                disp_vals.extend(per_example_means(
                    soft_divergence_per_token("soft", s_logp, u_logp),
                    counts))
        out: dict = {}
        if floor_vals:
            self.denom = sum(floor_vals) / len(floor_vals)
        if self.denom is not None:
            out["probe/noop_floor"] = self.denom
        if res_vals:
            res = sum(res_vals) / len(res_vals)
            disp = sum(disp_vals) / len(disp_vals)
            out.update({
                "probe/residual_kl": res,
                "probe/displacement_kl": disp,
                "probe/residual_ratio": res / max(self.denom, 1e-6),
                "probe/displacement_ratio": disp / max(self.denom, 1e-6),
            })
        return out

    @staticmethod
    def _set_boundaries(ctx_value, student_b) -> None:
        """A masked (decode-only) object's apply() yields a live
        SteeringHandle whose per-row boundary must be set from the batch's
        prompt lengths BEFORE the forward (never derived from the padded
        batch width — the trap). All-token objects yield None/other."""
        from ..hooks import SteeringHandle

        if (isinstance(ctx_value, SteeringHandle)
                and ctx_value.positions in ("response", "prompt")):
            ctx_value.set_prompt_lens(steer_prompt_boundaries(
                student_b, ctx_value.response_header_len))


# --------------------------------------------------------------------------
# Pre-clip global gradient norm (port of tune_lora_all's GradNormTap).
# --------------------------------------------------------------------------


class GradNormTap:
    """Pre-clip global gradient norm via per-parameter backward hooks —
    clipping mutates .grad in place before opt.step(), so the post-step
    .grad tensors are the CLIPPED gradients; the hooks see the raw ones.
    The hooks return None (observation only), so attaching a tap cannot
    change any gradient. The accumulator is a 0-dim GPU tensor;
    ``flush_tensor`` returns it WITHOUT a device sync (the path every
    consumer buffers through), ``flush`` converts (one sync — the
    convenience form for sync-tolerant callers)."""

    def __init__(self, params):
        self._sq = None
        self._handles = [p.register_hook(self._add) for p in params]

    def _add(self, g):
        v = g.detach().float().pow(2).sum()
        self._sq = v if self._sq is None else self._sq + v

    def flush_tensor(self) -> torch.Tensor | None:
        """The pre-clip global norm as a 0-dim tensor (no sync); ``None``
        when no backward has run since the last flush."""
        if self._sq is None:
            return None
        v = self._sq.sqrt()
        self._sq = None
        return v

    def flush(self) -> float:
        v = self.flush_tensor()
        return float("nan") if v is None else float(v)

    def remove(self):
        for h in self._handles:
            h.remove()


def postclip_grad_norm(params) -> torch.Tensor | None:
    """Global norm of the CURRENT .grad tensors as a 0-dim tensor (no
    device sync) — after ``clip_grad_norm_`` and before the next
    ``zero_grad()`` these are the post-clip gradients. ``None`` when no
    parameter carries a grad."""
    gs = [p.grad for p in params if p.grad is not None]
    if not gs:
        return None
    return torch.norm(torch.stack([g.detach().float().norm() for g in gs]))


# --------------------------------------------------------------------------
# Per-arm norm groups (the "monitor steering-vector norms" heuristic,
# generalized): lora |A|/|B| totals + per-layer |B_l|; psr per-layer
# gate_w/gate_b/direction norms + a parameter-shut gate fraction; vector
# norm; sentinel embedding norm.
# --------------------------------------------------------------------------


def norm_groups(obj) -> dict[str, torch.Tensor]:
    """Per-step object-norm metrics for any harness object, dispatched on
    the object's own attributes (never the arm name — bidpo_lora reuses
    lora's class and must inherit its reporter).

    SYNC-FREE by contract: every value is a 0-dim tensor on the object's
    device (``.norm()`` launches a kernel, never a sync); the caller
    ``.item()``s them at its logging cadence, not per step."""
    out: dict[str, torch.Tensor] = {}
    A = getattr(obj, "A", None)
    B = getattr(obj, "B", None)
    if isinstance(A, dict) and isinstance(B, dict):
        # MultiLayerLoRAObject: totals + per-layer |B_l| (the trained side). The factors are keyed by the object's OWN keys —
        # the layer int in the legacy o_proj shape (one factor per layer), ``"<layer>.<module>"`` in the module-set shape (FD 6:
        # several modules per block, routed experts as 3-D ``[E, r_e, d]`` factors) — never re-derived from ``obj.layers``: the
        # first module-set trial (paper-companion-repo stage E, 2026-09-30) died at its first heartbeat on ``A[0]`` (KeyError)
        # because ``obj.layers`` indexes a dict keyed by strings. Per layer, the modules' |B| are combined as one norm, so the
        # legacy layout's keys and values are unchanged (one factor -> that factor's norm).
        keys = list(getattr(obj, "factor_keys", None) or A.keys())
        if not keys:
            return out
        a_norms = torch.stack([A[k].detach().norm() for k in keys])
        b_norms = torch.stack([B[k].detach().norm() for k in keys])
        out["norms/A_total"] = a_norms.norm()
        out["norms/B_total"] = b_norms.norm()
        out["norms/B_max"] = b_norms.max()
        by_layer: dict[int, list[int]] = {}
        for i, k in enumerate(keys):
            layer = int(k) if isinstance(k, int) else int(str(k).split(".", 1)[0])
            by_layer.setdefault(layer, []).append(i)
        for layer in sorted(by_layer):
            idx = by_layer[layer]
            out[f"norms/B_{layer:02d}"] = b_norms[idx[0]] if len(idx) == 1 else torch.stack([b_norms[i] for i in idx]).norm()
        return out
    if isinstance(A, torch.nn.Parameter) and isinstance(B, torch.nn.Parameter):
        # RankOneLoRAObject (single layer).
        out["norms/A_total"] = A.detach().norm()
        out["norms/B_total"] = B.detach().norm()
        return out
    gate_w = getattr(obj, "gate_w", None)
    if isinstance(gate_w, dict):
        # PSRObject: per-layer gate_w/gate_b/direction norms + the
        # PARAMETER-shut gate fraction. A gate with gate_b <= 0 and
        # ||gate_w|| == 0 is provably ReLU-dead at every input; the
        # measured 23k-wave deaths froze at gate_b ~ 0.92 with small
        # nonzero |w| (input-dependent death), which parameters alone
        # cannot certify — the TRUE activity measure is
        # :func:`psr_gate_activity`, run at probe cadence.
        w = torch.stack([obj.gate_w[l].detach().norm() for l in obj.layers])
        b = torch.stack([obj.gate_b[l].detach().reshape(())
                         for l in obj.layers])
        d = torch.stack([obj.directions[l].detach().norm()
                         for l in obj.layers])
        for i, l in enumerate(obj.layers):
            out[f"norms/gate_w_{l:02d}"] = w[i]
            out[f"norms/gate_b_{l:02d}"] = b[i]
            out[f"norms/direction_{l:02d}"] = d[i]
        out["norms/direction_total"] = d.norm()
        out["norms/gate_b_min"] = b.min()
        out["norms/gate_param_dead_frac"] = (
            (b <= 0.0) & (w == 0.0)).float().mean()
        return out
    if hasattr(obj, "sentinel_id") and hasattr(obj, "param"):
        # SoftTokenObject / AppendedSoftTokenObject.
        out["norms/sentinel_embedding"] = obj.param.detach().norm()
        return out
    params_dict = getattr(obj, "params", None)
    if isinstance(params_dict, dict):
        # MultiLayerVectorObject.
        norms = torch.stack([params_dict[l].detach().norm()
                             for l in obj.layers])
        out["norms/vec_total"] = norms.norm()
        out["norms/vec_max"] = norms.max()
        return out
    trainable = list(getattr(obj, "trainable", []) or [])
    if len(trainable) == 1:
        out["norms/vec_norm"] = trainable[0].detach().norm()
        return out
    if trainable:
        out["norms/trainable_total"] = torch.norm(torch.stack(
            [p.detach().float().norm() for p in trainable]))
    return out


@torch.no_grad()
def psr_gate_activity(model, obj, student_b) -> dict[str, float]:
    """TRUE gate activity from one no-grad forward inside ``obj.apply``:
    the fraction of layers whose gate is zero at EVERY real position
    (``probe/gate_dead_frac``) and the mean over layers of the fraction of
    real positions with a live gate (``probe/gate_active_frac``). Reads
    the pre-steering gate stash (``PSRObject._last_gate``) INSIDE the
    context — it is cleared at exit. Run at probe cadence (one extra
    forward), never per step."""
    device = next(model.parameters()).device
    student_b = student_b.to(device)
    with obj.apply(model, scale=1.0) as ctx_value:
        KLRatioProbe._set_boundaries(ctx_value, student_b)
        model(input_ids=student_b.input_ids,
              attention_mask=student_b.attention_mask)
        real = student_b.attention_mask.bool()
        # One stacked device->host transfer for all layers (this runs at
        # probe cadence only, right after the probe's own sync).
        live = torch.stack(
            [(obj._last_gate[l][real] > 0).float().mean()
             if l in obj._last_gate else torch.zeros((), device=device)
             for l in obj.layers])
        fracs = live.tolist()
    n = max(len(obj.layers), 1)
    return {"probe/gate_dead_frac": sum(1 for f in fracs if f == 0.0) / n,
            "probe/gate_active_frac": sum(fracs) / n}


# --------------------------------------------------------------------------
# Per-cell metrics mirror (<cell-dir>/metrics.jsonl — the contract file).
# --------------------------------------------------------------------------


class CellMetricsWriter:
    """Append-only per-cell metrics jsonl. NUMERIC scalars only (the
    make_metrics_log_fn convention — text can never leave through this
    path), written and flushed per record so a crash cannot lose curves."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict, step=None) -> None:
        import json

        numeric = {k: v for k, v in record.items()
                   if isinstance(v, (int, float))
                   and not isinstance(v, bool)}
        if not numeric:
            return
        self._fh.write(json.dumps({"step": step, **numeric}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


# --------------------------------------------------------------------------
# The per-cell production monitor: probe + tap + cadence + norms + abort,
# behind the two trainer-facing callback shapes.
# --------------------------------------------------------------------------


class RatioDivergenceGuard:
    """Hard-stop on a sustained residual-ratio blow-up:
    ``probe/residual_ratio`` (= residual KL / the no-op floor) above
    ``threshold`` on ``consecutive`` CONSECUTIVE probe evals means the
    object is making the student markedly WORSE than no-op — terminate
    rather than burn the budget. Noise/warmup discipline (the
    patience-1 lesson): the first ``skip_probes`` probe evals never count
    (early-ramp transients), a single spike resets nothing by itself, and
    a healthy run (ratio < 1) never accumulates a strike. SEPARATE from
    and ADDITIVE to :class:`StepDivergenceGuard` (loss-scale explosions in
    step units); this one rules on the probed OBJECTIVE-level ratio.
    Non-finite ratios are ignored here — the callers' existing non-finite
    aborts own that case."""

    #: message prefix — callers classify the terminal status on it.
    PREFIX = "ratio-divergence"

    def __init__(self, threshold: float = 1.5, consecutive: int = 2,
                 skip_probes: int = 1):
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self.skip_probes = int(skip_probes)
        self.n_seen = 0
        self.strikes = 0

    def update(self, ratio) -> None:
        """Feed one probe eval's residual ratio; raises
        :class:`TrainingDiverged` on the ``consecutive``-th consecutive
        breach past the warmup skip."""
        import math as _math

        self.n_seen += 1
        if ratio is None or not _math.isfinite(float(ratio)):
            return
        if self.n_seen <= self.skip_probes:
            return
        if float(ratio) > self.threshold:
            self.strikes += 1
            if self.strikes >= self.consecutive:
                raise TrainingDiverged(
                    f"{self.PREFIX}: probe/residual_ratio="
                    f"{float(ratio):.3f} > {self.threshold} on "
                    f"{self.strikes} consecutive probe evals — the object "
                    "is worse than no-op; terminating")
        else:
            self.strikes = 0


class FitMonitor:
    """One production cell's monitoring bundle.

    Two callback shapes, matching the trainer seams:

    - ``on_step(step, obj, loss)`` — the lockstep ``probe_cb`` seam: per
      optimizer step, after ``opt.step()`` (grads still present, already
      clipped). Buffers ``train/loss``, ``train/grad_norm`` (pre-clip,
      from the attached :class:`GradNormTap`), ``train/grad_norm_postclip``
      and the object's norm groups AS 0-DIM TENSORS — no device sync —
      and flushes them to the log once per ``cadence.log_every`` steps
      (sync discipline, module docstring). At probe cadence it also runs
      the dual-KL probe (``probe/residual_ratio``,
      ``probe/displacement_ratio``, ``probe/heldout_loss`` = residual KL
      on the held-out probe) and, for gate objects, true gate activity.
    - ``on_probe(step, obj)`` — the solo-trainer ``probe_cb(step, obj)``
      seam (train_bidpo): cadence-gated dual-KL probe + norms only
      (the trainer logs its own per-step loss/grad telemetry through
      ``log_fn``).

    ANY non-finite metric raises :class:`TrainingDiverged` at the flush —
    the abort the caller converts into a failed arm (detection latency
    <= ``log_every`` steps; NaNs only propagate in between). Emission
    goes through ``emit(record, step)`` (the driver wires it to the
    per-cell metrics.jsonl + the shared log stream). ``heartbeat``
    (optional): a :class:`Heartbeat` fed per step from the loss the
    trainer already synced — no extra sync."""

    def __init__(self, arm: str, model, probe: KLRatioProbe,
                 cadence: ProbeCadence, emit, *, gate_batch=None,
                 heartbeat=None, ratio_guard=None):
        self.arm = arm
        self.model = model
        self.probe = probe
        self.cadence = cadence
        self._emit = emit
        self._gate_batch = gate_batch
        self.heartbeat = heartbeat
        #: default ON: the residual-ratio hard-stop (pass None to disable
        #: — e.g. a probe-free routing test with a stub engine).
        self.ratio_guard = (RatioDivergenceGuard() if ratio_guard is None
                            else (None if ratio_guard is False
                                  else ratio_guard))
        self.tap: GradNormTap | None = None
        self.last_step: int = 0
        #: How many optimizer steps this monitor OBSERVED (a stubbed
        #: engine drives none — the audit keys its per-step expectations
        #: on this, so a routing test with a fake trainer still publishes).
        self.steps_seen: int = 0
        self._buf: list[tuple[int, dict]] = []

    def attach(self, obj) -> "FitMonitor":
        """Arm the pre-clip gradient tap on ``obj.trainable`` (lockstep
        arms, whose object exists before the fit)."""
        self.tap = GradNormTap(obj.trainable)
        return self

    def close(self) -> None:
        self.flush()
        if self.tap is not None:
            self.tap.remove()
            self.tap = None

    # -- lockstep seam ------------------------------------------------------
    def on_step(self, step: int, obj, loss) -> None:
        self.last_step = step
        self.steps_seen += 1
        rec: dict = {"train/loss": loss}
        if self.tap is not None:
            pre = self.tap.flush_tensor()
            if pre is not None:
                rec["train/grad_norm"] = pre
        post = postclip_grad_norm(getattr(obj, "trainable", []) or [])
        if post is not None:
            rec["train/grad_norm_postclip"] = post
        rec.update(norm_groups(obj))
        self._buf.append((step, rec))
        if self.heartbeat is not None and isinstance(loss, (int, float)):
            self.heartbeat.step(step, float(loss), None)
        probe_due = self.cadence.due(step)
        if probe_due or self.cadence.log_due(step):
            extra = None
            if probe_due:
                if self.heartbeat is not None:
                    self.heartbeat.phase(
                        f"{self.arm}: dual-KL probe at step {step} "
                        "(no-step pass — training resumes after)")
                extra = self._probe(obj)
            self.flush(extra=extra, extra_step=step)

    # -- solo-trainer seam (probe_cb(step, obj) — train_bidpo) --------------
    def on_probe(self, step: int, obj) -> None:
        self.last_step = step
        if not self.cadence.due(step):
            return
        rec = dict(self._probe(obj))
        rec.update(norm_groups(obj))
        self._buf.append((step, rec))
        self.flush()

    # -- flush: the ONE sync point per logging interval ----------------------
    def flush(self, extra: dict | None = None,
              extra_step: int | None = None) -> None:
        """Materialise every buffered record (tensor -> float in ONE
        stacked device sync for the whole backlog — a per-value ``float()``
        here would be log_every x n_metrics separate D2H round-trips per
        flush, the sync discipline's own failure mode), emit them in step
        order, then run the non-finite abort over everything just
        materialised."""
        buffered, self._buf = self._buf, []
        if extra is not None:
            buffered.append((extra_step if extra_step is not None
                             else self.last_step, dict(extra)))
        tensors = [v for _step, rec in buffered for v in rec.values()
                   if isinstance(v, torch.Tensor)]
        flat = iter(torch.stack([t.detach().float().reshape(())
                                 for t in tensors]).tolist()
                    if tensors else ())  # ONE sync
        problems: list[str] = []
        for step, rec in buffered:
            vals = {k: (next(flat) if isinstance(v, torch.Tensor) else v)
                    for k, v in rec.items()}
            if self._emit is not None:
                self._emit(vals, step)
            try:
                assert_finite(vals, step=step, context=self.arm)
            except TrainingDiverged as e:
                problems.append(str(e))
        if problems:
            raise TrainingDiverged("; ".join(problems))

    # -- end of fit ----------------------------------------------------------
    def final(self, obj) -> dict[str, float]:
        """The final dual-KL measurement the pre-publication audit
        requires for every cell, emitted under ``final/`` keys and
        returned for the fit record."""
        self.flush()  # any straggling per-step records first
        kl = self.probe.measure(self.model, obj)
        rec = {("final/" + k.split("/", 1)[1]): v for k, v in kl.items()}
        self.flush(extra=rec, extra_step=self.last_step)
        return rec

    def _probe(self, obj) -> dict[str, float]:
        kl = dict(self.probe.measure(self.model, obj))
        if self.ratio_guard is not None:
            # Raises TrainingDiverged through the caller's probe seam —
            # the same terminal path as the step guard.
            self.ratio_guard.update(kl.get("probe/residual_ratio"))
        if "probe/residual_kl" in kl:
            # The held-out loss at probe cadence: the residual KL IS the
            # held-out distillation loss on the probe subset.
            kl["probe/heldout_loss"] = kl["probe/residual_kl"]
        if (self._gate_batch is not None
                and isinstance(getattr(obj, "gate_w", None), dict)):
            kl.update(psr_gate_activity(self.model, obj, self._gate_batch))
        return kl


# --------------------------------------------------------------------------
# Continuous heartbeat (training must log continuously — nobody should have
# to infer a phase from nvidia-smi) and production length-bucketing, both
# ported from tune_lora_all @ 8ded0ea.
# --------------------------------------------------------------------------


class GpuDutySampler:
    """Continuous GPU SM-utilisation sampling for the fit loop (gpu-util
    skill: DUTY is the metric — instantaneous ``nvidia-smi`` lies for
    bursty loads, and a spot-check burst cannot tell a stall from a quiet
    phase). One ``nvidia-smi dmon -s u -d 1`` subprocess streams 1-second
    samples for the process lifetime; a daemon thread parses them into
    ``(t, sm%)``. ``window(since)`` gives the mean/min over an interval
    (the heartbeat's line), ``summary()`` the whole run (the bench's
    table). Pinned to THIS process's device: the first entry of
    ``CUDA_VISIBLE_DEVICES`` when set (a shared node's honor-system
    isolation makes a bare dmon report a neighbour's duty as ours), else
    device 0 (a Modal / single-GPU container). Disabled — every method a
    no-op returning None — when CUDA or ``nvidia-smi`` is absent, so CPU
    tests and cluster login nodes never spawn anything."""

    def __init__(self, enabled: bool | None = None):
        import shutil
        import threading

        if enabled is None:
            try:
                import torch
                enabled = bool(torch.cuda.is_available()) \
                    and shutil.which("nvidia-smi") is not None
            except Exception:  # noqa: BLE001 — no torch, no GPU sampling
                enabled = False
        self.enabled = bool(enabled)
        self.samples: list[tuple[float, int]] = []
        self._proc = None
        self._lock = threading.Lock()
        if self.enabled:
            self._start()

    @staticmethod
    def device_index() -> str:
        import os

        vis = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
        return vis[0].strip() if vis and vis[0].strip() else "0"

    def _start(self) -> None:
        import atexit
        import subprocess
        import threading

        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-i",
                 self.device_index()],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.enabled = False
            self._proc = None
            return
        atexit.register(self.close)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            parts = line.split()
            # "# gpu sm mem enc dec ..." headers repeat; data rows start
            # with the device index and carry sm% second ('-' when N/A).
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            try:
                sm = int(parts[1])
            except ValueError:
                continue
            with self._lock:
                self.samples.append((time.time(), sm))

    def window(self, since_t: float) -> dict | None:
        """``{"mean", "min", "max", "n"}`` over samples taken after
        ``since_t``; None when disabled or empty."""
        if not self.enabled:
            return None
        with self._lock:
            vals = [sm for t, sm in self.samples if t >= since_t]
        if not vals:
            return None
        return {"mean": round(sum(vals) / len(vals), 1), "min": min(vals),
                "max": max(vals), "n": len(vals)}

    def summary(self) -> dict | None:
        return self.window(0.0)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass


class Heartbeat:
    """Continuous liveness logging. A daemon thread prints one line every
    ``interval`` seconds: the named pre-training phase before step 0, then
    step/total, loss, LR, s/step EMA, tok/s and GPU SM% over the interval
    (:class:`GpuDutySampler`) and ETA — with an explicit note when the fit
    is inside a long no-step pass (dense KL probe, per-epoch held-out
    eval), so a quiet GPU never reads as a stall. Feed it from values the
    trainer has ALREADY synced (its own float loss, its pre-computed token
    counts) — the heartbeat itself must not add device syncs. ``tokens``
    per step are optional: trainers without a token record print no tok/s.

    Every instance is also held WEAKLY in :attr:`_LIVE` so :meth:`close_all` can stop the
    threads a caller never closed: a fit script that REFUSES (SystemExit) after building its
    heartbeat and before its ``hb.close()`` leaves the daemon thread printing ``[hb]`` lines
    every ``interval`` seconds for the life of the process. In production the process ends
    with the refusal; under pytest the process is the whole suite, and the stray lines land
    in whichever later test has stdout captured — a byte-identity test over a captured
    dry run failed on exactly that in the cc1f0e7d joint suite (2026-09-12).
    ``tests/conftest.py`` closes the leaked ones after every test."""

    #: weak set of every heartbeat whose thread may still be running (see the docstring)
    _LIVE = None

    def __init__(self, total_steps: int, interval: float = 30.0,
                 gpu_duty: bool | None = None):
        import threading
        import weakref

        self.t0 = time.time()
        self.total = total_steps
        self.state = {"phase": "init", "phase_t": self.t0, "step": None,
                      "loss": float("nan"), "lr": None, "spe": None,
                      "last_step_t": self.t0, "tokens": 0,
                      "tokens_seen": False,
                      "line_t": self.t0, "line_tokens": 0}
        self.duty = GpuDutySampler(enabled=gpu_duty)
        self._stop = threading.Event()
        self._interval = interval
        self._thread = threading.Thread(target=self._run, daemon=True)
        if Heartbeat._LIVE is None:
            Heartbeat._LIVE = weakref.WeakSet()
        Heartbeat._LIVE.add(self)
        self._thread.start()

    @classmethod
    def live(cls) -> list:
        """The heartbeats whose printer thread is still running (a running thread keeps its
        instance alive through the bound ``_run`` target, so the weak set holds exactly those)."""
        return [] if cls._LIVE is None else [hb for hb in list(cls._LIVE) if hb._thread.is_alive()]

    @classmethod
    def close_all(cls, join_s: float = 2.0) -> int:
        """Stop every live heartbeat (a caller that exited before its ``close()``); returns how
        many were still running. Joins each thread briefly so no ``[hb]`` line can print after
        this returns."""
        live = cls.live()
        for hb in live:
            hb.close()
        for hb in live:
            hb._thread.join(timeout=join_s)
        return len(live)

    def phase(self, msg: str) -> None:
        self.state.update(phase=msg, phase_t=time.time())
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"[phase +{time.time() - self.t0:.0f}s] {msg}", flush=True)

    def step(self, step: int, loss: float, lr, tokens: int | None = None
             ) -> None:
        now = time.time()
        dt = now - self.state["last_step_t"]
        spe = self.state["spe"]
        self.state.update(step=step, loss=loss, lr=lr, last_step_t=now,
                          spe=dt if spe is None else 0.9 * spe + 0.1 * dt)
        if tokens is not None:
            self.state["tokens"] += int(tokens)
            self.state["tokens_seen"] = True

    def _line(self) -> None:
        s = self.state
        now = time.time()
        if s["step"] is None:
            print(f"[{time.strftime('%H:%M:%S')}] [hb] pre-training: "
                  f"{s['phase']} ({now - s['phase_t']:.0f}s in "
                  f"phase, {now - self.t0:.0f}s total)", flush=True)
            return
        gap = now - s["last_step_t"]
        spe = s["spe"] or 0.0
        eta = (self.total - s["step"]) * spe / 60 if spe else float("nan")
        note = (f" | {gap:.0f}s since last step (probe/held-out pass)"
                if gap > max(15.0, 5 * spe) else "")
        lr = f" lr={s['lr']:.2e}" if s["lr"] is not None else ""
        # Interval rates: tokens and SM% since the previous heartbeat line.
        tps = ""
        if s["tokens_seen"]:
            dtok = s["tokens"] - s["line_tokens"]
            dt_line = max(now - s["line_t"], 1e-9)
            tps = f" tok/s={dtok / dt_line:,.0f}"
        duty = self.duty.window(s["line_t"])
        sm = (f" sm={duty['mean']:.0f}% (min {duty['min']}%)"
              if duty else "")
        s["line_t"], s["line_tokens"] = now, s["tokens"]
        print(f"[{time.strftime('%H:%M:%S')}] [hb] step "
              f"{s['step']}/{self.total} loss={s['loss']:.4f}{lr} "
              f"{spe:.2f}s/step{tps}{sm} eta={eta:.0f}min{note}", flush=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._line()

    def duty_summary(self) -> dict | None:
        """Whole-run SM% statistics from the sampler (None when disabled)."""
        return self.duty.summary()

    def close(self) -> None:
        self._stop.set()
        self.duty.close()


def length_sorted(tok, examples):
    """Length-bucket batching for the PRODUCTION fit path (measured with
    cuDNN-SDPA-off at 2.6 -> 0.62-0.78 s/step and 33-45% -> 99-100% GPU
    duty at identical batch size on the lora r5 confirm wave): the engines
    chunk examples IN THE ORDER GIVEN (and shuffle only batch ORDER per
    epoch), so sorting by rendered token length makes every batch pad to a
    near-uniform width. Two wins: ragged batches wasted ~30-40% of token
    compute on padding, and every distinct padded shape made cuDNN SDPA
    re-plan on CPU (also fixed by :func:`pin_attention_backends`). Teacher
    order follows student order by construction (one example = one aligned
    pair).

    NOTE: batch COMPOSITION changes vs an unsorted fit — same objective,
    different gradient grouping; every cell that trains bucketed must say
    so in its provenance (``length_bucketed`` in the sidecar). The densest
    batch becomes all-long rows, so memory probes must measure the
    N-LONGEST rows, not a random sample (gpu_memprobe --worst-case)."""
    from .tokcache import cached_ids

    def keylen(ex):
        return (len(cached_ids(tok, ex.prompt_without, True))
                + len(cached_ids(tok, ex.completion, False)))

    return sorted(examples, key=keylen)
