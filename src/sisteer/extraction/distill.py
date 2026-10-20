"""Distillation of a system instruction into a steering vector.

The objective that literally earns the word "compile": make the
UNPROMPTED, steered model match the SI-PROMPTED teacher (same frozen
weights) on teacher-generated completions.

- loss="soft": token-level KL( p(.|SI, x, y_<t) || p_steered(.|x, y_<t) )
  over completion positions (forward KL, teacher-forced on the teacher's
  own completions). Preserves the whole conditional distribution, unlike
  pairwise preference losses.
- loss="hard": cross-entropy of the steered student on the teacher
  completions — exactly HyperSteer's Eq. 4 with a fixed per-instruction
  vector instead of a hypernetwork.
- loss="rkl": reverse KL( p_steered || p(.|SI, ...) ) on the same
  positions — mode-seeking where forward KL is mass-covering, so the
  student may concentrate on teacher modes instead of spreading over
  the whole (possibly partly non-compliant) teacher distribution.
- loss="js": Jensen-Shannon divergence — the symmetric, bounded (log 2)
  midpoint between the two KL directions.

Prompt and completion are tokenized separately and concatenated at the
token level, so the completion's token ids are identical in the teacher
context (with SI) and the student context (without) — the alignment the
KL depends on.

The same engine takes any vector provider (a fixed nn.Parameter or a
hypernetwork), so HyperSteer-NC reuses it (see hypersteer.py).
"""

from __future__ import annotations

import math
import time
import random
import warnings
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ..hooks import steering
from ..models import hidden_size
from ..vectors import SteeringVector
from .prefix_cache import TeacherPrefixCache, prefix_cache_for_batches
from .traindata import Batch, collate, EncodedSeq
from .tokcache import cached_ids


@dataclass
class DistillExample:
    prompt_with_si: str  # teacher prompt (system = the instruction)
    prompt_without: str  # student prompt (no instruction)
    completion: str  # teacher-generated completion
    instruction_idx: int = 0  # used by the hypernet variant


@dataclass(frozen=True)
class MaxLenPreflight:
    """What preflight_max_len measured (returned on success so entry
    points can log it / derive configs from it)."""

    longest_prompt_tokens: int  # max over examples of max(teacher, student)
    longest_example_idx: int  # which example carried it
    longest_side: str  # "teacher" or "student"
    required_max_len: int  # longest + min_completion_tokens


def preflight_max_len(
    tok,
    examples: list[DistillExample],
    max_len: int,
    *,
    min_completion_tokens: int = 1,
) -> MaxLenPreflight:
    """Refuse a doomed training run BEFORE any GPU work.

    Tokenizes every rendered teacher and student prompt — exactly the
    strings _encode_aligned will tokenize, with the same
    add_special_tokens=True — and checks that max_len holds the longest
    of them plus at least min_completion_tokens of completion budget
    (the EOS _encode_aligned appends counts toward that budget). Pure
    CPU: only the tokenizer runs, so entry points can call it before the
    model is even loaded.

    At the default min_completion_tokens=1 this is exactly
    _encode_aligned's own ``avail < 1`` refusal, so preflight-pass
    implies encoding cannot raise; a larger value additionally
    guarantees that many supervision tokens survive truncation for
    EVERY example (pass completion_cap + 1 to keep a full completion
    plus its EOS).

    Why it exists: with the inherited stage-2 ``max_len: 768``,
    SAFETY_SI's ~129-token teacher prompt fits easily, but a
    multi-thousand-token SI (the 8-section CONSTITUTION_SI renders to
    ~5,908 tokens before the attack text) makes _encode_aligned raise
    its terse "max_len too small for prompts" on the FIRST batch of
    every run — after the expensive pair pool was generated. This turns
    that into an up-front refusal that names the numbers and the fix.
    """
    if min_completion_tokens < 1:
        raise ValueError(
            "min_completion_tokens must be >= 1 — an example whose "
            "completion is entirely truncated away has nothing to "
            "supervise (and _encode_aligned refuses it)"
        )
    if not examples:
        raise ValueError("preflight_max_len needs at least one example")
    longest, longest_idx, longest_side = -1, -1, ""
    for i, ex in enumerate(examples):
        t_len = len(cached_ids(tok, ex.prompt_with_si, True))
        s_len = len(cached_ids(tok, ex.prompt_without, True))
        n, side = (t_len, "teacher") if t_len >= s_len else (s_len, "student")
        if n > longest:
            longest, longest_idx, longest_side = n, i, side
    required = longest + min_completion_tokens
    if max_len < required:
        raise ValueError(
            f"max_len={max_len} cannot hold these prompts: the longest "
            f"rendered {longest_side} prompt (example {longest_idx} of "
            f"{len(examples)}) is {longest} tokens, and a completion "
            f"budget of at least {min_completion_tokens} token(s) (EOS "
            f"included) must fit after it, so max_len must be >= "
            f"{required}. Set max_len >= {required} in the training "
            f"config — never inherit a short-SI max_len for a long "
            f"system instruction (see configs/two_si.yaml for the "
            f"derivation)."
        )
    return MaxLenPreflight(
        longest_prompt_tokens=longest,
        longest_example_idx=longest_idx,
        longest_side=longest_side,
        required_max_len=required,
    )


class TokenizeProgress:
    """Progress for the tokenization pass.

    This is the phase between model load and step 1, and on a long-SI pool it
    is minutes to half an hour: a 20,660-pair pool at max_len 8192 took ~31
    min per cell, and the phase emitted NOTHING while it ran. Eight cells
    looked hung simultaneously and were read as normal startup three times
    before anyone measured it. A silent slow phase is indistinguishable from
    a wedged one, so it gets a line.

    Cheap by construction: a counter and a clock, reported at most ~10 times
    plus a final line, so it cannot itself become the cost. Printing does not
    touch the encode, so the bitwise contract is unaffected.
    """

    def __init__(self, total: int, *, label: str, max_len: int):
        self.total = int(total)
        self.label = label
        self.max_len = int(max_len)
        self._t0 = time.monotonic()
        self._step = max(1, self.total // 10)
        self._last = 0
        if self.total:
            print(f"[tokenize] {label}: {self.total} examples at max_len "
                  f"{self.max_len} — no token cache on this branch, so this "
                  f"runs per fit", flush=True)

    def tick(self, done: int) -> None:
        if not self.total or done - self._last < self._step or done >= self.total:
            return
        self._last = done
        el = time.monotonic() - self._t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float("nan")
        print(f"[tokenize] {self.label}: {done}/{self.total} "
              f"({100.0 * done / self.total:.0f}%) {el:.0f}s elapsed, "
              f"~{eta:.0f}s left", flush=True)

    def done(self) -> None:
        if not self.total:
            return
        el = time.monotonic() - self._t0
        print(f"[tokenize] {self.label}: {self.total} examples in {el:.0f}s "
              f"({self.total / el:.0f}/s)" if el > 0 else
              f"[tokenize] {self.label}: {self.total} examples", flush=True)


def _encode_aligned(
    tok, ex: DistillExample, max_len: int
) -> tuple[EncodedSeq, EncodedSeq]:
    comp = cached_ids(tok, ex.completion, False)
    if tok.eos_token_id is not None:
        comp = comp + [tok.eos_token_id]
    t_prompt = cached_ids(tok, ex.prompt_with_si, True)
    s_prompt = cached_ids(tok, ex.prompt_without, True)
    avail = max_len - max(len(t_prompt), len(s_prompt))
    if avail < 1:
        raise ValueError("max_len too small for prompts")
    comp = comp[:avail]
    return (
        EncodedSeq(ids=t_prompt + comp, prompt_len=len(t_prompt)),
        EncodedSeq(ids=s_prompt + comp, prompt_len=len(s_prompt)),
    )


def _gather_completion_logits(
    logits: torch.Tensor, batch: Batch, gather: torch.Tensor | None = None
) -> torch.Tensor:
    """Logits at positions predicting completion tokens → [N_comp_tokens, V].

    ``gather`` (optional [B, S-1] bool) overrides the default completion
    mask — the mask→loss wiring (see :func:`loss_gather_masks`): when the
    steering hook steers a SUBSET of positions, the loss must read exactly
    the completion positions that subset covers, from the SAME mask object
    the hook used. ``None`` keeps the historical behavior bit-for-bit."""
    mask = gather if gather is not None else (batch.labels[:, 1:] != -100)
    return logits[:, :-1][mask]


def steer_prompt_boundaries(
    batch: Batch, response_header_len: int | None
) -> torch.Tensor:
    """[B] per-row steering boundary for a full-sequence TRAINING forward:
    each row's prompt length minus the trailing assistant-header allowance
    (the header-steered convention: the H header tokens are steered, so the
    logits that pick the FIRST completion token come from steered hidden
    states). ``None``/0 is the PLAIN decode-only boundary (= prompt width).
    Clamped at 0 for degenerate prompts shorter than the header.

    This is the training-forward counterpart of the handle's own per-prefill
    derivation (``max(seq_len - H, 0)``): a training forward is prompt +
    completion + padding wide, so its ``seq_len`` is NOT the prompt width —
    deriving the boundary from it silently steers only the last H columns
    of the padded batch. Always pass THESE boundaries to
    ``handle.set_prompt_lens`` before the forward (the trap
    ``loss_gather_masks`` raises on)."""
    h = int(response_header_len or 0)
    return (batch.prompt_lens - h).clamp(min=0)


def loss_gather_masks(
    handle, student_b: Batch, teacher_b: Batch | None = None
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """The mask→loss wiring: boolean gather masks selecting the
    completion-predicting logit positions the loss may read, derived from
    THE SAME mask the steering hook applied during the student forward.

    ``handle`` is the live :class:`~sisteer.hooks.SteeringHandle` of the
    student forward, or ``None`` when the intervention exposes no position
    mask (weight edits, soft tokens, PSR hooks — which apply at every
    position, i.e. the all-token convention). The mask comes from
    ``handle._position_mask`` — the single place a position mask is
    computed (hooks.py) — never recomputed here, so hook and loss cannot
    disagree.

    Returns ``(s_gather, t_gather, keep)``:

    - ``s_gather`` [B, S_s-1]: completion positions ∩ steered positions
      (the logit at position j is computed from the — possibly steered —
      hidden state at j, so the hook mask is consumed at ``[:, :-1]``);
    - ``keep`` [N_comp] | None: the kept flag per completion token in the
      shared row-major flatten order — ``None`` when nothing is dropped,
      so default paths stay bit-identical;
    - ``t_gather`` [B, S_t-1] | None: keeps exactly the completion indices
      the student kept, mapped to the teacher's own (longer, SI-bearing)
      coordinates. The teacher is never steered; its mask exists only so
      the two gathers stay position-aligned.

    Fail-loudly guards (this repo's convention — a wrong mask must never
    silently shrink the loss):

    - a row whose supervised ∩ steered set is EMPTY raises (the
      unset-boundary trap: ``positions='response'`` with ``prompt_len``
      left at 0, or a boundary derived from the padded batch width);
    - under the header-steered convention (``handle.response_header_len``
      set) every completion-predicting logit position is steered BY
      CONSTRUCTION (boundary = prompt_len - H <= first completion logit
      position), so ANY dropped completion position means the boundary was
      misderived and raises rather than training on the wrong support.
    """
    s_keep = student_b.labels[:, 1:] != -100
    t_keep = None if teacher_b is None else (teacher_b.labels[:, 1:] != -100)
    pos_mask = None
    if handle is not None:
        b, s = student_b.input_ids.shape
        pos_mask = handle._position_mask(b, s, student_b.input_ids.device)
    if pos_mask is None:
        return s_keep, t_keep, None
    s_gather = s_keep & pos_mask[:, :-1].bool()
    if not bool(s_gather.any(dim=1).all()):
        row = int((~s_gather.any(dim=1)).long().argmax())
        raise ValueError(
            f"mask→loss wiring: row {row} has NO completion position inside "
            "the steering mask — nothing to supervise. This is the "
            "unset/misderived-boundary trap (positions='response' with "
            "prompt_len left at its default, or a boundary derived from the "
            "PADDED batch width): set per-row boundaries via "
            "handle.set_prompt_lens(steer_prompt_boundaries(batch, H)) "
            "before the training forward."
        )
    if torch.equal(s_gather, s_keep):
        return s_keep, t_keep, None
    if handle.response_header_len is not None:
        raise ValueError(
            "mask→loss wiring: the header-steered mask dropped supervised "
            "completion positions. Under boundary = prompt_len - H every "
            "completion-predicting logit position is steered by "
            "construction, so a dropped position means the boundary was "
            "misderived (the padded batch width minus H instead of each "
            "row's prompt_len minus H?)."
        )
    keep = s_gather[s_keep]
    t_gather = None
    if t_keep is not None:
        t_gather = t_keep.clone()
        t_gather[t_keep] = keep
    return s_gather, t_gather, keep


# The teacher-distribution targets ("hard" is the odd one out: CE on the
# teacher's tokens needs no teacher forward pass).
SOFT_LOSSES = ("soft", "rkl", "js")


def soft_divergence(loss: str, t_logp: torch.Tensor,
                    s_logp: torch.Tensor) -> torch.Tensor:
    """The distribution-matching objective on [N, V] completion-position
    log-probs (teacher, student), reduced batchmean over N:

    - "soft": forward KL(t || s) — mass-covering;
    - "rkl":  reverse KL(s || t) — mode-seeking (F.kl_div backpropagates
      through the target argument, which is where the student sits here);
    - "js":   JS(t, s) = KL(t || m)/2 + KL(s || m)/2 with the mixture
      midpoint m computed in log space.
    """
    if loss == "soft":
        return F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean")
    if loss == "rkl":
        return F.kl_div(t_logp, s_logp, log_target=True, reduction="batchmean")
    if loss == "js":
        m_logp = torch.logaddexp(t_logp, s_logp) - math.log(2.0)
        return 0.5 * (
            F.kl_div(m_logp, t_logp, log_target=True, reduction="batchmean")
            + F.kl_div(m_logp, s_logp, log_target=True, reduction="batchmean")
        )
    raise ValueError(f"unknown soft divergence {loss!r} (known: {SOFT_LOSSES})")


def soft_divergence_per_token(loss: str, t_logp: torch.Tensor,
                              s_logp: torch.Tensor) -> torch.Tensor:
    """Per-token soft divergence on [N, V] completion-position log-probs —
    the UNREDUCED counterpart of :func:`soft_divergence`, returning [N].
    Pinned identity (tests): ``soft_divergence_per_token(...).mean() ==
    soft_divergence(...)`` for every loss in ``SOFT_LOSSES``, because
    ``reduction='batchmean'`` on [N, V] is exactly the mean over N of the
    per-token sum over V.

    Exists for the probe/floor ESTIMATORS (never the training step): a
    probe that averages per-BATCH batchmeans weighs tokens by which ragged
    chunk they landed in, so the estimate moves when ``batch_size`` — a
    memory knob — changes. Per-token values grouped back per EXAMPLE
    (:func:`per_example_means`) make the estimate a mean over examples,
    invariant to chunking."""
    if loss == "soft":
        return F.kl_div(s_logp, t_logp, log_target=True,
                        reduction="none").sum(-1)
    if loss == "rkl":
        return F.kl_div(t_logp, s_logp, log_target=True,
                        reduction="none").sum(-1)
    if loss == "js":
        m_logp = torch.logaddexp(t_logp, s_logp) - math.log(2.0)
        return 0.5 * (
            F.kl_div(m_logp, t_logp, log_target=True,
                     reduction="none").sum(-1)
            + F.kl_div(m_logp, s_logp, log_target=True,
                       reduction="none").sum(-1)
        )
    raise ValueError(f"unknown soft divergence {loss!r} (known: {SOFT_LOSSES})")


def soft_divergence_topk(t_idx: torch.Tensor, t_val: torch.Tensor,
                         s_logits: torch.Tensor, *,
                         temperature: float = 1.0) -> torch.Tensor:
    """Forward KL against a PRECOMPUTED top-K teacher head — the
    offline-teacher counterpart of ``soft_divergence("soft", ...)``
    (:mod:`.teacher_topk` builds the artifact; the wave consumes it via
    ``train_lockstep(teacher_topk=...)``).

    THE EXACT FORM (documented because the HPO tuned every LR on the
    full-vocab online loss, and any change of objective must be stated,
    measured, and gated — never silent). With ``v`` the stored top-K
    teacher logits at indices ``t_idx``, ``z`` the student logits over the
    full vocab, and ``T`` the temperature::

        q_k       = softmax(v / T)_k            k = 1..K   (renormalized head)
        log s_k   = log_softmax(z / T)[t_idx_k]            (FULL-vocab student)
        L         = (1/N) * sum_n sum_k  q_k (log q_k - log s_k)

    ``softmax`` over a SUBSET of logits IS the full softmax restricted to
    that subset and renormalized (q_k = t_k / Z with Z = the head's true
    mass), so this is KL(q || s) with the teacher's top-K mass renormalized
    to 1 over its own support while the student stays the full-vocab
    log-softmax gathered at those K indices. Relative to the full-vocab
    forward KL it differs by (a) the 1/Z scaling of the head terms plus the
    student-independent constant ``-log Z`` (no gradient), and (b) the
    dropped tail ``sum_{v not in head} t_v (log t_v - log s_v)``; its
    student-logit gradient is ``(softmax(z/T) - q·1_head) / T`` versus the
    full loss's ``(softmax(z/T) - t) / T`` — identical up to ``q`` vs ``t``
    on the head (a factor 1/Z) and the missing ``-t_v`` pull on tail
    tokens (total tail mass ``1 - Z``), so it converges to the full
    gradient as the head mass Z -> 1. At T=1 on a trained production teacher Z is ~1 and
    the two losses agree to the tolerance the verification step reports;
    at T=10 (token_faithful) the head mass collapses (top-K by logit is
    temperature-invariant as a SET, but the mass it covers is not) and
    this form is NOT the tuned loss — consumers must refuse it there
    (train_lockstep does).

    FORWARD KL ONLY: rkl/js need teacher-vs-student mass over the
    student's whole support, which a top-K head cannot carry — a
    reverse-KL "approximation" from a head would silently be a different
    objective, so there is deliberately no ``loss`` parameter here.

    Reduction is ``batchmean`` over the N gathered positions — identical
    to the full-vocab path's ``F.kl_div(..., reduction="batchmean")`` on
    ``[N, V]``, so histories and floor ratios stay commensurable."""
    t_logp = torch.log_softmax(t_val.float() / temperature, -1)
    s_logp = torch.log_softmax(s_logits.float() / temperature, -1)
    s_logp_head = s_logp.gather(-1, t_idx.long())
    return F.kl_div(s_logp_head, t_logp, log_target=True,
                    reduction="batchmean")


def per_example_means(per_token: torch.Tensor, counts) -> list[float]:
    """Group a flat [N] per-token value vector back into per-EXAMPLE means.

    ``counts`` is the per-row kept-token count ([B] tensor or list) of the
    gather that produced the flat vector; boolean-mask gathering flattens
    row-major, so consecutive slices of length ``counts[i]`` are example
    ``i``'s tokens. Rows with count 0 are skipped (nothing was measured for
    them — never a fake 0.0). The per-example token-mean is the chunking-
    invariant unit every probe/floor estimator averages over."""
    if torch.is_tensor(counts):
        counts = [int(c) for c in counts.tolist()]
    else:
        counts = [int(c) for c in counts]
    total = sum(counts)
    if total != int(per_token.numel()):
        raise ValueError(
            f"per_example_means: counts sum to {total} but the per-token "
            f"vector has {int(per_token.numel())} entries — the gather and "
            "the counts disagree (different masks?)")
    out: list[float] = []
    off = 0
    for k in counts:
        if k > 0:
            out.append(float(per_token[off:off + k].mean()))
        off += k
    return out


# How many examples a probe averages over by default. 256 is EMPIRICALLY
# justified, not asserted. Two measurements, and their difference is the
# whole lesson:
#
# The FIRST n=256 measurement (job 187947) took the POOL PREFIX, and the
# pool is sorted by (family, goal_id, variant_key), so its "256 paired
# prompts" were 235 aim + 21 amplegcg_plus — ONE attack family that is 1%
# of the pool. Its floors (0.10516 / 0.07129 / 0.06898) were SINGLE-FAMILY
# figures: "KL on aim prompts", not per-SI teacher signal. Only its CI
# width (+/-7-9% of the mean at n=256) survives as the resolution evidence
# this default rests on.
#
# The RE-MEASURE (cluster B job 30542, 2026-08-24: sha256(prompt)-rank
# within every family stratum, 96/family -> 1037 paired identities per SI
# over all 11 families, per-example estimator, this module's primitives;
# full per-family table + selected ids in
# exp/twosi_floors/floors_stratified_30542.json) gives the
# pool-representative (pool-weighted) floors
#   safety_refusal   0.05579  95% CI [0.04938, 0.06219]   (-47% vs prefix)
#   si_constitution  0.11035  95% CI [0.09731, 0.12340]   (+55% vs prefix)
#   si_irrelevant    0.10125  95% CI [0.08837, 0.11412]   (+47% vs prefix)
# The prefix floors are recovered as the aim stratum of the re-measure
# (0.114 / 0.070 / 0.060 — each holding the old number inside its CI), so
# the contamination diagnosis is confirmed quantitatively, and the old
# between-SI ordering (safety_refusal highest) was an artifact: it is
# LOWEST. Family heterogeneity dominates: the equal-family pooled floor is
# ~2x the pool-weighted one on every SI, and benign-stratum floors
# (phtest + wildjailbreak_benign, 50% of the pool) run 2.5-5.8x below
# harmful-stratum ones — so no scalar floor should travel without its
# per-family table (ProbeEstimate.per_family / the measurement json).
DEFAULT_PROBE_N = 256


def _mean_sd_sem(vals: list[float]) -> tuple[float, float | None,
                                             float | None]:
    """Unweighted mean of PER-EXAMPLE probe values, their sample sd, and
    the sem of the mean (``sd/sqrt(n_examples)``). Deliberately the SAME
    estimator as ``scripts/measure_teacher_signal.py`` (the measured true
    floors), so probe numbers and measured floors are directly comparable
    — both moved together from per-batch batchmeans to per-example means
    (2026-08-24): a mean of per-batch scalars over ragged token counts
    changes with ``batch_size``, a memory knob, which defeated decoupling
    statistics from memory width. sd/sem are ``None`` below two values:
    with one example the spread is unresolvable, and a ``None`` is a
    stated absence where a 0.0 would claim false certainty."""
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, None, None
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    sd = math.sqrt(var)
    return m, sd, sd / math.sqrt(len(vals))


def _ratio_sem(acts: list[float], noops: list[float]) -> float | None:
    """Delta-method sem of ``mean(acts) / mean(noops)`` — the floor ratio's
    own uncertainty, so a guard (or a reader applying a stricter criterion,
    e.g. ratio < 1.0) can tell a resolvable ratio from noise. The two legs
    are measured on the SAME probe examples, so their per-example errors
    correlate; the covariance term keeps that from inflating the sem:

        Var(A/F) ~ (Var(A) + R^2 Var(F) - 2 R Cov(A,F)) / F^2

    with A, F the per-example-mean estimates and R = A/F. ``None`` when
    the spread is unresolvable (fewer than two examples, mismatched legs,
    or a non-positive floor)."""
    b = len(acts)
    if b < 2 or len(noops) != b:
        return None
    a_m = sum(acts) / b
    f_m = sum(noops) / b
    if f_m <= 0:
        return None
    var_a = sum((x - a_m) ** 2 for x in acts) / (b - 1) / b
    var_f = sum((x - f_m) ** 2 for x in noops) / (b - 1) / b
    cov = sum((x - a_m) * (y - f_m)
              for x, y in zip(acts, noops)) / (b - 1) / b
    r = a_m / f_m
    var_r = (var_a + r * r * var_f - 2.0 * r * cov) / (f_m * f_m)
    return math.sqrt(max(var_r, 0.0))


class ProbeEstimate(tuple):
    """``(active, noop)`` — still a 2-tuple, so every existing caller's
    ``act, floor = arm_probe_losses(...)`` unpack keeps working bit-for-bit
    (``lofit.heldout_probe`` and the fit driver both unpack two floats) —
    that ADDITIONALLY carries the estimate's provenance and uncertainty:

    - ``active`` / ``noop``: the means (items 0 and 1), over PER-EXAMPLE
      token-mean divergences — invariant to ``batch_size``, which chunks
      the forwards for memory and nothing else;
    - ``active_sd``/``active_sem``, ``noop_sd``/``noop_sem``: per-example
      spread and the sem of each mean (``None`` below two examples);
    - ``ratio`` / ``ratio_sem``: ``active / max(noop, 1e-6)`` (the
      recorded floor-ratio convention) and its delta-method sem;
    - ``active_values`` / ``noop_values``: the per-example values
      themselves (tuples, selection order), so a caller can re-pair legs
      measured in separate passes or re-group them by strata;
    - ``n_examples`` / ``n_batches`` / ``batch_size``: how many examples
      the estimate REALLY used and how they were chunked — so a recorded
      number can never again silently be a 2-sample estimate;
    - ``selected_sha8``: 8-hex-char sha256 prefixes of the selection keys
      (the student prompt renderings), selection order — the population a
      recorded floor describes, re-derivable from the pool;
    - ``per_family``: optional ``{family: {active, noop, ratio, n, ...}}``
      breakdown when the caller supplied family labels — per-family fit
      ratios are the number commensurable with per-family attack-success
      rates (a pooled ratio averages that structure away irreversibly).
    """

    def __new__(cls, active: float, noop: float, *,
                active_sd: float | None = None,
                active_sem: float | None = None,
                noop_sd: float | None = None,
                noop_sem: float | None = None,
                ratio_sem: float | None = None,
                n_examples: int = 0, n_batches: int = 0,
                batch_size: int = 0,
                active_values: tuple[float, ...] | None = None,
                noop_values: tuple[float, ...] | None = None,
                selected_sha8: tuple[str, ...] | None = None,
                per_family: dict | None = None):
        self = super().__new__(cls, (float(active), float(noop)))
        self.active = float(active)
        self.noop = float(noop)
        self.active_sd = active_sd
        self.active_sem = active_sem
        self.noop_sd = noop_sd
        self.noop_sem = noop_sem
        self.ratio_sem = ratio_sem
        self.n_examples = int(n_examples)
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.active_values = active_values
        self.noop_values = noop_values
        self.selected_sha8 = selected_sha8
        self.per_family = per_family
        return self

    @property
    def ratio(self) -> float:
        return self.active / max(self.noop, 1e-6)


# --------------------------------------------------------------------------- #
# ProbeEstimate -> fit-record persistence contract                             #
# --------------------------------------------------------------------------- #
# Every field ProbeEstimate carries, mapped to the key its value is persisted
# under in a fit record (train_steered_lm's diagnostics, train_lockstep's
# per-arm diagnostics, the driver's per-arm fit_record — and through them
# every sidecar and fit report they feed). The map is the WRITE-SIDE
# CONTRACT: a field named here must reach the record under that key at every
# site that writes one. History (2026-08-24): the writers were built flat,
# so every SCALAR field survived and every STRUCTURED one (arrays, dicts)
# was silently dropped — the per-example values existed at write time and
# were deleted, and recovering them meant RE-PROBING finished objects on a
# GPU; a paired cross-cell comparison (same probe examples, correlated
# errors) then had to overstate its difference sem by up to ~3x with the
# independence formula. tests/test_probe_values_record.py introspects
# ProbeEstimate's constructor and fails — naming the field — on any field
# in neither this map nor PROBE_ESTIMATE_UNPERSISTED, so the next structured
# field cannot vanish the same way. (General diagnostic that would have
# caught this in seconds: when auditing any writer, ask whether it drops
# everything non-scalar.)
# NOTE on sources, not keys: a record may fill noop_floor* from the STARTUP
# estimate and final_probe_* from the END-OF-TRAINING one (distill/lockstep
# do); the key each field lands under is invariant across sites.
PROBE_ESTIMATE_RECORD_KEYS: dict[str, str] = {
    "active": "final_probe_loss",
    "noop": "noop_floor",
    "active_sd": "final_probe_sd",
    "active_sem": "final_probe_sem",
    "noop_sd": "noop_floor_sd",
    "noop_sem": "noop_floor_sem",
    "ratio_sem": "floor_ratio_sem",
    "n_examples": "probe_n",
    "n_batches": "probe_n_batches",
    "batch_size": "probe_batch_size",
    "active_values": "probe_active_values",
    "noop_values": "probe_noop_values",
    "selected_sha8": "probe_selected_sha8",
    "per_family": "probe_per_family",
}

# Fields deliberately NOT persisted, each with its stated reason. Empty on
# purpose today: everything the estimator computes is worth keeping (the
# arrays are ~10 KB at the 256-example default). An entry here must name
# WHY, so an exclusion is a decision on record, never an oversight.
PROBE_ESTIMATE_UNPERSISTED: dict[str, str] = {}

# WRITER-LEVEL exemptions — the field map above binds the RECORD-WRITING
# sites only, and an empty field-level exclusion set must not be read as
# "everything is persisted everywhere": some emitters legitimately never
# carry the per-example payload. Each is named here with its reason, so the
# decline is on record the same way a field exclusion would be. HOW THIS
# LIST (and the site list) was derived, reproducibly: enumerate the code
# paths that OBTAIN per-example probe values — callers of arm_probe_losses
# and probe_steered_lm, plus train_lockstep's _probe_pair lists — and follow
# each value to every write. Never by grepping persisted key names, which
# has exactly this blind spot one level down (an emitter that never wrote
# the keys never matches the grep).
PROBE_EMISSION_POINTS_EXEMPT: dict[str, str] = {
    "floor_ratio_check.log_fn": (
        "receives final_loss/noop_floor as floats, never holds a "
        "ProbeEstimate — structurally cannot carry arrays; metrics-stream "
        "scalars only"),
    "startup_loss_check.log_fn": (
        "same shape: two floats in, warn-or-not out; the caller holding "
        "the estimate writes the record"),
    "live log_fn records (train_steered_lm / train_lockstep / the driver)": (
        "per-step metrics streams (W&B/jsonl) are scalar time series by "
        "design, and the driver's metrics path filters to numbers "
        "regardless; the same values reach the sidecar via diagnostics"),
    "lofit/reft per-epoch heldout_probe emitters + best_heldout_probe": (
        "per-EPOCH early-stop metric holding unpacked floats; per-epoch "
        "arrays would multiply size with no pairing value — those arms' "
        "end-of-fit record carries the arrays via the driver's solo path"),
}


def _round_sig7(v):
    """None-safe rounding to 7 significant digits (float32-ish). The
    per-example arrays are a probe record's bulk, and full float64 reprs
    (~17 sig digits) would triple their bytes for precision no analysis of
    a ~1e-2-sem probe can use. Non-finite values pass through unchanged --
    json emits NaN/Infinity tokens for them either way."""
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return v
    return float(f"{v:.7g}")


def probe_values_record(*, active_values=None, noop_values=None,
                        per_family=None, prefix: str = "probe") -> dict:
    """The PER-EXAMPLE payload of a probe record: the structured
    ProbeEstimate fields (see PROBE_ESTIMATE_RECORD_KEYS) that every flat
    scalar writer used to drop. Additive by design — callers ``update``
    an existing record with this dict; no existing key is renamed, moved
    or removed. ``prefix`` swaps the key family (``heldout_probe`` for the
    held-out leg).

    Values are rounded to 7 significant digits (:func:`_round_sig7`);
    ``per_family`` travels verbatim (a per-family table is a handful of
    aggregate scalars, like the record's other unrounded aggregates).
    ``selected_sha8`` is NOT emitted here — it is the cross-cell alignment
    key and must stay verbatim under the sites' existing
    ``probe_selected_sha8`` convention (``list(sha8 or ())``).

    Size: bounded by the probe's realised sample size (``probe_n``, default
    ``DEFAULT_PROBE_N`` = 256, clamped to the pool) — two float arrays plus
    the sha8 list is roughly 35 bytes/example, ~9 KB at the default. With
    ``n_probe=None`` the probe covers the whole fit pool, so the arrays are
    pool-sized; that is deliberate and uncapped, because truncating them
    would silently break the example pairing these fields exist to provide.

    Absence is graceful, never fatal: a missing leg writes ``None``, an
    empty leg writes ``[]`` — a fit that trained for hours must never die
    in its writer."""
    def _vals(vs):
        return None if vs is None else [_round_sig7(v) for v in vs]

    return {
        f"{prefix}_active_values": _vals(active_values),
        f"{prefix}_noop_values": _vals(noop_values),
        f"{prefix}_per_family": per_family,
    }


def probe_family_table(families: list[str], acts: list[float],
                       noops: list[float] | None) -> dict:
    """Per-family (active, noop, ratio, n) breakdown of paired per-example
    probe values — the grouping that makes a fit ratio commensurable with
    per-family attack-success rates. A bare floor is a HEADROOM measure
    (how much the SI changes behaviour on that family with no object in
    play); the per-family RATIO divides that headroom out. Means carry
    their sem when resolvable (n >= 2); a thin family reports its point
    estimate WITH its n and ``None`` sems — visible thinness, never a
    number that looks solid and is two samples."""
    if len(families) != len(acts) or (noops is not None
                                      and len(noops) != len(acts)):
        raise ValueError(
            f"probe_family_table: {len(families)} family labels for "
            f"{len(acts)} active / "
            f"{'-' if noops is None else len(noops)} noop values — the "
            "labels must be parallel to the per-example values")
    table: dict[str, dict] = {}
    for fam in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == fam]
        a = [acts[i] for i in idx]
        a_m, _sd, a_sem = _mean_sd_sem(a)
        row = {"active": a_m, "active_sem": a_sem, "n": len(idx)}
        if noops is not None:
            f_vals = [noops[i] for i in idx]
            f_m, _fsd, f_sem = _mean_sd_sem(f_vals)
            row.update({"noop": f_m, "noop_sem": f_sem,
                        "ratio": a_m / max(f_m, 1e-6),
                        "ratio_sem": _ratio_sem(a, f_vals)})
        table[fam] = row
    return table


def startup_loss_check(step0_loss: float, noop_loss: float, *,
                       tag: str, log_fn=None) -> None:
    """Startup sanity check: record the as-initialized (step-0) loss next to
    the SAME batch's loss with the object INACTIVE — the no-op floor — and
    warn PROMINENTLY when step-0 starts far above it. A zero-init object
    starts exactly AT the floor (ratio 1); informed inits (token_faithful,
    the single-layer psr_kl diffmean) legitimately start somewhat above it,
    so this warns rather than raises. The bar is an ORDER OF MAGNITUDE over
    the floor — a ratio, because ``loss_scale`` multiplies both sides (so
    the check is scale-invariant) and because it is calibrated to the
    failure it exists to catch, not to typical informed inits: the
    diffmean-init all-layer PSR started at KL 9.28 against a 0.148 floor
    (63x) and the first AdamW step killed its deep-layer gates — then
    nobody noticed for 966 steps. The 1e-6 floor guard only matters when
    the no-op reference is essentially zero (teacher == student), where any
    visible step-0 loss is infinitely far above it multiplicatively."""
    if log_fn:
        log_fn({f"{tag}/step0_loss": step0_loss,
                f"{tag}/noop_floor": noop_loss}, 0)
    if step0_loss > 10.0 * max(noop_loss, 1e-6):
        warnings.warn(
            f"{tag}: step-0 loss {step0_loss:.4f} is far above the no-op "
            f"floor {noop_loss:.4f} (same batch, object inactive). The "
            "object starts far off the unsteered model — a bad init can "
            "saturate ReLU gates shut in a single optimizer step and then "
            "train to nothing. Check the arm's init.",
            RuntimeWarning, stacklevel=2)


class FloorRatioError(RuntimeError):
    """End-of-training loss stuck above ratio_max × the no-op floor."""


def floor_ratio_check(final_loss: float, noop_floor: float, *,
                      ratio_max: float, tag: str, log_fn=None,
                      min_floor: float = 1e-2) -> float:
    """End-of-training counterpart of :func:`startup_loss_check` — and this
    one RAISES rather than warns. Refuse any arm whose trained loss never
    came within ``ratio_max`` (default 3, config-supplied) of the measured
    no-op floor: same-batch convention, so ``final_loss`` is the probe
    batch's loss with the TRAINED object active and ``noop_floor`` the same
    batch with it inactive. The ratio is scale-invariant (``loss_scale``
    multiplies both sides). Calibration: the real failure this catches
    plateaued at KL 4.82 against a 0.148 floor — ~30x worse than doing
    nothing — for 966 silent steps. Returns the ratio when it passes.

    ``min_floor`` is a MATERIALITY clamp on the denominator: when the
    measured floor is essentially zero (teacher ≈ student on the probe
    batch — tiny fixtures, heavily benign pools), any visible trained loss
    is "infinitely" far above it multiplicatively while being an excellent
    fit in absolute terms (a batchmean KL of 2e-4 is not a degenerate
    object). Clamping at 1e-2 keeps the guard about the failure it is
    calibrated to (multiples of a MATERIAL floor like 0.148) instead of
    ratio noise around zero; the 4.82-vs-0.148 case still fails at ~32x."""
    if not (math.isfinite(final_loss) and math.isfinite(noop_floor)):
        # A NaN/inf probe must REFUSE, not pass: every comparison against a
        # NaN is False, so \"ratio > ratio_max\" would wave a NaN'd fit
        # through as if it cleared the bar — and for arms without a
        # dead-gate guard (vector, lora_r1) nothing downstream would stop
        # the save (observed live: psr_kl_all's response-leg fit went NaN
        # in epoch 1 on 2026-08-25; its gate guard caught it, a vector
        # arm's would not have been).
        raise FloorRatioError(
            f"{tag}: non-finite floor probe (final_loss={final_loss!r}, "
            f"noop_floor={noop_floor!r}) — the trained object or the probe "
            "diverged; refusing to treat a NaN as a passing ratio.")
    ratio = final_loss / max(noop_floor, min_floor)
    if log_fn:
        log_fn({f"{tag}/final_probe_loss": final_loss,
                f"{tag}/noop_floor": noop_floor,
                f"{tag}/floor_ratio": ratio})
    if ratio > ratio_max:
        raise FloorRatioError(
            f"{tag}: end-of-training loss {final_loss:.4f} is {ratio:.1f}x "
            f"the no-op floor {noop_floor:.4f} (max allowed {ratio_max}x, "
            "same probe batch, object active vs inactive). The trained "
            "object leaves the unprompted student no closer to — or further "
            "from — the SI-prompted teacher than doing nothing; a "
            "degenerate fit must fail loudly, never reach disk."
        )
    return ratio


@dataclass(frozen=True)
class EarlyStopSpec:
    """Per-arm early stopping on a held-out loss curve. ``update`` runs
    once per completed epoch; training stops after ``patience`` consecutive
    non-improving epochs (once ``min_epochs`` have run), and the weights
    kept are the BEST epoch's, not the last (``restore_best``)."""

    patience: int = 2
    min_epochs: int = 1
    min_delta: float = 0.0


class EarlyStopper:
    """The stateful side of :class:`EarlyStopSpec` — one per arm, so
    different arms stop at different epochs (the point: token_bos was still
    improving at epoch 6 while psr_mse_all had converged by ~3)."""

    def __init__(self, spec: EarlyStopSpec):
        self.spec = spec
        self.best = math.inf
        self.best_epoch = 0  # 1-based completed-epoch count of the best eval
        self.epochs_seen = 0
        self._since_best = 0
        self._snapshot: list[torch.Tensor] | None = None

    def update(self, heldout_loss: float, params) -> bool:
        """Record one completed epoch's held-out loss; snapshot ``params``
        on a new best; return True when training should stop."""
        self.epochs_seen += 1
        if heldout_loss < self.best - self.spec.min_delta:
            self.best = float(heldout_loss)
            self.best_epoch = self.epochs_seen
            self._snapshot = [p.detach().clone() for p in params]
            self._since_best = 0
        else:
            self._since_best += 1
        return (self.epochs_seen >= self.spec.min_epochs
                and self._since_best >= self.spec.patience)

    def observe_exempt(self) -> None:
        """Count one completed epoch WITHOUT ruling on it: no strike, no new
        best, no snapshot — only ``epochs_seen`` advances, so the counter
        keeps matching the trainer's epoch index (the checkpoint's
        internal-consistency invariant, ``continuation.load_state``) and
        ``min_epochs`` still counts the epoch as run. For the REPLAYED
        remainder of a continuation's checkpoint epoch: a partial epoch
        trained under a freshly restarted lr is not evidence about the fit,
        so it neither spends patience nor competes for the best-epoch
        snapshot; the first FULL post-restart epoch is where the count
        resumes."""
        self.epochs_seen += 1

    def restore_best(self, params) -> None:
        """Copy the best epoch's snapshot back into ``params`` (no-op when
        no epoch was recorded). Deliberately NEVER restores the pre-training
        init: an arm that only got worse keeps its (bad) best-trained-epoch
        weights so the floor-ratio guard fails it LOUDLY, instead of a
        restored no-op init slipping past the guard looking healthy."""
        if self._snapshot is None:
            return
        with torch.no_grad():
            for p, snap in zip(params, self._snapshot):
                p.data.copy_(snap)


def probe_steered_lm(
    model,
    tok,
    examples: list[DistillExample],
    layer: int,
    vector_provider: Callable[[list[int]], torch.Tensor],
    *,
    loss: str = "soft",
    temperature: float = 1.0,
    max_len: int = 768,
    batch_size: int = 4,
    positions: str = "all",
    skip_bos: bool = True,
    response_header_len: int | None = None,
    n_probe: int | None = DEFAULT_PROBE_N,
) -> ProbeEstimate:
    """The (active, no-op) probe for the vector-provider trainers
    (:func:`train_steered_lm` / :func:`train_distill` / hypersteer) — on
    the SAME estimator the lockstep path and the fit driver rule with.

    This replaces the second, parallel guard implementation that still
    read ``batches[0]``: ``train_distill`` / ``train_steered_lm`` called
    outside the lockstep driver kept ruling their startup warning and
    end-of-training ``floor_ratio_check`` on a 2-4 sample batchmean —
    exactly the hazard the lockstep path removed. Selection, per-example
    accumulation, uncertainty and the recorded population all come from
    ONE estimator now:

    - probe examples are chosen by sha256-rank of the student rendering
      over the WHOLE pool (``subsample.sha_rank_indices`` — never a
      prefix of a family-sorted pool), then grouped by ``instruction_idx``
      so each group probes under its own vector (the hypernet case);
    - the soft losses route through ``lockstep.arm_probe_losses`` via a
      minimal wrapper object (inject = identity, apply = the SAME
      ``hooks.steering`` ctx the training step uses, vector fetched from
      ``vector_provider`` at probe time so the trained vector is what is
      probed) — deliberately CALLING the shared estimator rather than
      transcribing it;
    - ``loss='hard'`` keeps a small parallel leg here by necessity: hard
      consumes teacher TOKENS (cross-entropy against targets), which the
      teacher-divergence estimator has no leg for. It uses the same
      selection, the same per-example unit (per-token CE grouped by the
      gather's row counts) and the same statistics, and its chunk
      invariance is pinned by test alongside the soft path's equality
      with ``arm_probe_losses``.
    """
    if loss != "hard" and loss not in SOFT_LOSSES:
        raise ValueError(f"loss must be 'hard' or one of {SOFT_LOSSES}")
    if not examples:
        raise ValueError("probe_steered_lm needs at least one example")
    from .subsample import sha_hex, sha_rank_indices

    keys = [ex.prompt_without for ex in examples]
    sel_idx = sha_rank_indices(keys, n_probe)
    selected = [examples[i] for i in sel_idx]
    groups: dict[int, list[DistillExample]] = {}
    for ex in selected:
        groups.setdefault(ex.instruction_idx, []).append(ex)

    device = next(model.parameters()).device
    bs = max(min(int(batch_size), max(len(g) for g in groups.values())), 1)
    acts: list[float] = []
    noops: list[float] = []
    sha8: list[str] = []
    n_batches = 0

    if loss != "hard":
        from .lockstep import LockstepArmParams, arm_probe_losses

        class _ProbeVectorObject:
            """The harness object protocol over a vector_provider group —
            the probe-side twin of lockstep's _VectorArmObject."""

            def __init__(self, instruction_idx: int):
                self._idx = instruction_idx

            def inject(self, rendered_prompt: str) -> str:
                return rendered_prompt

            def apply(self, m, scale: float = 1.0):
                vec = vector_provider([self._idx])
                return steering(m, {layer: vec}, scale=scale,
                                positions=positions, skip_bos=skip_bos,
                                response_header_len=response_header_len)

        for idx, group in groups.items():
            est = arm_probe_losses(
                model, tok, _ProbeVectorObject(idx), group,
                max_len=max_len, batch_size=bs, n_probe=None,
                params=LockstepArmParams(loss=loss,
                                         temperature=temperature))
            acts.extend(est.active_values)
            noops.extend(est.noop_values)
            sha8.extend(est.selected_sha8)
            n_batches += est.n_batches
    else:
        for idx, group in groups.items():
            encoded = [_encode_aligned(tok, ex, max_len) for ex in group]
            for i in range(0, len(encoded), bs):
                chunk = encoded[i:i + bs]
                student_b = collate([s for _, s in chunk],
                                    tok.pad_token_id).to(device)
                with torch.no_grad():
                    vec = vector_provider([idx])
                    with steering(model, {layer: vec}, positions=positions,
                                  skip_bos=skip_bos,
                                  response_header_len=response_header_len
                                  ) as handle:
                        if positions in ("response", "prompt"):
                            handle.set_prompt_lens(steer_prompt_boundaries(
                                student_b, response_header_len))
                        act_logits = model(
                            input_ids=student_b.input_ids,
                            attention_mask=student_b.attention_mask).logits
                        s_gather, _tg, _keep = loss_gather_masks(handle,
                                                                 student_b)
                    inact_logits = model(
                        input_ids=student_b.input_ids,
                        attention_mask=student_b.attention_mask).logits
                    targets = student_b.labels[:, 1:][s_gather]
                    counts = s_gather.sum(1)

                    def _leg(logits):
                        s_logits = _gather_completion_logits(
                            logits, student_b, gather=s_gather)
                        per_tok = F.cross_entropy(s_logits.float(), targets,
                                                  reduction="none")
                        return per_example_means(per_tok, counts)

                    acts.extend(_leg(act_logits))
                    noops.extend(_leg(inact_logits))
                n_batches += 1
            sha8.extend(sha_hex(ex.prompt_without)[:8] for ex in group)

    act_m, act_sd, act_sem = _mean_sd_sem(acts)
    noop_m, noop_sd, noop_sem = _mean_sd_sem(noops)
    return ProbeEstimate(act_m, noop_m,
                         active_sd=act_sd, active_sem=act_sem,
                         noop_sd=noop_sd, noop_sem=noop_sem,
                         ratio_sem=_ratio_sem(acts, noops),
                         n_examples=len(selected), n_batches=n_batches,
                         batch_size=bs,
                         active_values=tuple(acts), noop_values=tuple(noops),
                         selected_sha8=tuple(sha8))


def train_steered_lm(
    model,
    tok,
    examples: list[DistillExample],
    layer: int,
    vector_provider: Callable[[list[int]], torch.Tensor],
    trainable_params: list[nn.Parameter],
    *,
    loss: str = "soft",
    lr: float = 5e-3,
    weight_decay: float = 0.01,
    epochs: int = 3,
    batch_size: int = 4,
    max_len: int = 768,
    temperature: float = 1.0,
    positions: str = "all",
    skip_bos: bool = True,
    response_header_len: int | None = None,
    seed: int = 0,
    log_fn=None,
    teacher_logits_hook: Callable[[int, Batch], torch.Tensor] | None = None,
    heldout_examples: list[DistillExample] | None = None,
    early_stop: EarlyStopSpec | None = None,
    floor_ratio_max: float | None = None,
    diagnostics: dict | None = None,
    n_probe: int | None = DEFAULT_PROBE_N,
    teacher_prefix_cache: bool = False,
) -> list[float]:
    """Shared engine. vector_provider(instruction_idxs) -> [d] vector for the
    batch (batches are grouped by instruction, so one vector per batch).
    Returns the loss history.

    ``teacher_prefix_cache``: every teacher prompt of one instruction
    shares its rendered prefix (chat header + system=SI), so under the flag
    the prefix runs through the model ONCE per instruction group and the
    GRADIENT-TRAINING teacher forwards (the step loop's, when no
    ``teacher_logits_hook`` supplies them) replay it as past_key_values
    (:mod:`.prefix_cache`). The prefix is derived from the FIT batches
    only, per instruction, matching lockstep's derivation — so a
    lockstep-vs-solo comparison under the same flag stays bitwise.
    DECISION-BEARING passes are AUTHORITATIVE AND UNCACHED regardless of
    the flag: the startup/final probe (whose numbers ``floor_ratio_check``
    rules on — a rejection writes no artifact, so a cache-flipped verdict
    is unrecoverable) and the held-out pass (which selects the restored
    epoch). Semantically exact, fp-tolerance-only difference on the
    training legs — see prefix_cache.py; the two-SI config's
    ``teacher_prefix_cache: false`` is the escape hatch. Default ``False``
    here keeps every other caller byte-identical. Teacher legs only: the
    student forward, which carries the gradients, is untouched.

    ``teacher_logits_hook`` (opt. 4 — lockstep teacher sharing): when given,
    the SI-prompted teacher's completion-position logits are OBTAINED from the
    hook — ``teacher_logits_hook(batch_index, teacher_batch) -> [N_comp, V]`` —
    instead of being computed by a local teacher forward. The teacher is the
    frozen base with the intervention INACTIVE, so its forward is bit-identical
    for every arm on the same (SI, batch); the hook lets ONE forward feed many
    arms rather than recomputing it per arm. It is a pure interposition on
    WHERE the (already-gathered) teacher logits come from — the loss semantics,
    the ``no_grad`` teacher context (the hook must honour it), and every other
    operation are unchanged, so with the hook returning exactly what the local
    forward would have produced the result is BITWISE identical. When the hook
    is ``None`` this engine behaves byte-for-byte as before (the ``hard`` loss
    never consumes teacher logits, so the hook is ignored there).

    Mask→loss wiring: the steering ctx's OWN handle drives both the hook
    and the loss positions. For a boundary mode (``positions='response'`` /
    ``'prompt'``) each batch's per-row boundary is set from the student
    batch's prompt lengths (minus ``response_header_len`` for the
    header-steered convention) via ``set_prompt_lens`` — never derived from
    the padded batch width — and the loss gathers exactly the completion
    positions that mask covers (:func:`loss_gather_masks`), with the teacher
    gather kept index-aligned. The default all-token leg is bit-identical to
    the historical behavior.

    ``heldout_examples`` + ``early_stop``: per-arm early stopping. The
    held-out loss (same objective, object active, no grad) is evaluated
    after every epoch; training stops per :class:`EarlyStopSpec` and the
    BEST epoch's params are restored. ``floor_ratio_max``: end-of-training
    no-op-floor guard (:func:`floor_ratio_check`) — raises
    :class:`FloorRatioError` when the trained probe loss exceeds
    ``floor_ratio_max`` x the same examples' object-inactive loss. The
    probe is :func:`probe_steered_lm` over up to ``n_probe`` sha-rank-
    selected examples (per-example estimator, selection recorded) — it
    was batch 0 only, i.e. every guard verdict here was a 2-4 sample
    batchmean of a family-sorted pool's first family.
    ``diagnostics`` (optional dict) is filled with the probe numbers with
    their sd/sem/counts/selection, the held-out curve, and the chosen
    stopping epoch, for the caller's report.
    """
    if loss != "hard" and loss not in SOFT_LOSSES:
        raise ValueError(f"loss must be 'hard' or one of {SOFT_LOSSES}")
    # every distillation path funnels through this engine, so the
    # max_len preflight here cannot be skipped by accident. It runs
    # before the model is touched (CPU tokenizer only) and, on failure,
    # names the offending prompt length and the minimum workable
    # max_len instead of _encode_aligned's terse error. The default
    # budget of 1 mirrors _encode_aligned exactly — no previously-valid
    # run is refused.
    preflight_max_len(tok, examples, max_len)
    if heldout_examples:
        preflight_max_len(tok, heldout_examples, max_len)
    device = next(model.parameters()).device
    rng = random.Random(seed)
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    # weight_decay exposed 2026-08-27 (re-tune): AdamW had silently been
    # applying its own default (0.01) to every trained object with nothing
    # tuning or recording it. The default HERE is that same 0.01, so every
    # caller that does not set it stays byte-identical.
    opt = torch.optim.AdamW(trainable_params, lr=lr,
                            weight_decay=weight_decay)

    # Group by instruction so a batch shares one steering vector.
    def _encode_grouped(exs: list[DistillExample]) -> list[tuple[int, Batch, Batch]]:
        by_instr: dict[int, list[DistillExample]] = {}
        for ex in exs:
            by_instr.setdefault(ex.instruction_idx, []).append(ex)
        out: list[tuple[int, Batch, Batch]] = []
        for idx, group in by_instr.items():
            encoded = [_encode_aligned(tok, ex, max_len) for ex in group]
            for i in range(0, len(encoded), batch_size):
                chunk = encoded[i : i + batch_size]
                teacher = collate([t for t, _ in chunk], tok.pad_token_id)
                student = collate([s for _, s in chunk], tok.pad_token_id)
                out.append((idx, teacher, student))
        return out

    batches = _encode_grouped(examples)
    heldout_batches = (
        _encode_grouped(heldout_examples) if heldout_examples else []
    )

    # Teacher prefix caches, one per instruction group, derived from that
    # group's FIT teacher batches only (fit-only derivation is what keeps a
    # solo run bitwise-comparable to lockstep's). GRADIENT-TRAINING
    # forwards only: the probe and held-out passes below are
    # decision-bearing and never touch these caches, and with a
    # teacher_logits_hook no local training forward exists either.
    tpc_by_idx: dict[int, TeacherPrefixCache] = {}
    if (teacher_prefix_cache and loss in SOFT_LOSSES
            and teacher_logits_hook is None):
        fit_teacher_by_idx: dict[int, list[Batch]] = {}
        for idx, teacher_b, _student_b in batches:
            fit_teacher_by_idx.setdefault(idx, []).append(teacher_b)
        for idx, tbs in fit_teacher_by_idx.items():
            tpc = prefix_cache_for_batches(model, tbs)
            if tpc is not None:
                tpc_by_idx[idx] = tpc

    def _uncached_teacher_logits(teacher_b: Batch) -> torch.Tensor:
        """The plain full forward + completion gather — the AUTHORITATIVE
        assembly for the decision-bearing held-out pass (the probe rides
        the shared estimator, :func:`probe_steered_lm`, which has no cache
        path of its own). Deliberately ignores the prefix caches: a guard
        verdict or a restored epoch must never depend on cached-kernel fp
        drift."""
        with torch.no_grad():
            return _gather_completion_logits(
                model(input_ids=teacher_b.input_ids,
                      attention_mask=teacher_b.attention_mask).logits,
                teacher_b,
            )

    def _local_teacher_logits(idx: int, teacher_b: Batch) -> torch.Tensor:
        """The completion-gathered teacher logits for one TRAINING batch —
        prefix-cached when this instruction group has a cache (fp-tolerance
        exact; the cache re-derives the usable prefix per batch and falls
        back to the identical full forward when nothing matches). Never
        used by the probe or held-out passes."""
        tpc = tpc_by_idx.get(idx)
        if tpc is not None:
            return tpc.completion_logits(model, teacher_b)
        return _uncached_teacher_logits(teacher_b)

    def _student_pass(student_b: Batch, vec):
        """One student forward under the steering ctx; the loss gather is
        derived from THE SAME handle the hook consumed (grad mode is the
        caller's)."""
        with steering(
            model, {layer: vec}, positions=positions, skip_bos=skip_bos,
            response_header_len=response_header_len,
        ) as handle:
            if positions in ("response", "prompt"):
                handle.set_prompt_lens(
                    steer_prompt_boundaries(student_b, response_header_len))
            logits = model(
                input_ids=student_b.input_ids,
                attention_mask=student_b.attention_mask,
            ).logits
            s_gather, _t_gather, keep = loss_gather_masks(handle, student_b)
        return logits, s_gather, keep

    def _teacher_logits(bi: int, idx: int, teacher_b: Batch,
                        keep) -> torch.Tensor:
        if teacher_logits_hook is not None:
            # Lockstep teacher sharing (opt. 4): the gathered
            # completion-position teacher logits are supplied from a
            # single shared forward. Byte-identical to the local path
            # below when the hook returns the same tensor.
            t_logits = teacher_logits_hook(bi, teacher_b)
        else:
            t_logits = _local_teacher_logits(idx, teacher_b)
        if keep is not None and not bool(keep.all()):
            t_logits = t_logits[keep]
        return t_logits

    def _probe() -> ProbeEstimate:
        """The shared-estimator probe (:func:`probe_steered_lm`) over up
        to ``n_probe`` sha-rank-selected examples — never batch 0, which
        was a 2-4 sample batchmean of a family-sorted pool's first family
        (review findings 1-3). The teacher forward is recomputed locally
        rather than through ``teacher_logits_hook``: the base is frozen,
        so the hook's contract makes the tensors identical either way.
        AUTHORITATIVE AND UNCACHED by construction (review finding
        2026-08-24): the shared estimator has no ``teacher_prefix_cache``
        path at all, so the numbers startup_loss_check and
        floor_ratio_check rule on never read cached kernels, flag or no
        flag — pinned by the cache-call-count tests in
        tests/test_prefix_cache.py."""
        return probe_steered_lm(
            model, tok, examples, layer, vector_provider, loss=loss,
            temperature=temperature, max_len=max_len,
            batch_size=batch_size, positions=positions, skip_bos=skip_bos,
            response_header_len=response_header_len, n_probe=n_probe)

    def _heldout_loss() -> float:
        """Position-weighted mean held-out loss (object active, no grad),
        under the SAME mask convention as training. The teacher is a local
        forward — the lockstep hook indexes TRAIN batches only."""
        tot, n = 0.0, 0
        with torch.no_grad():
            for idx, teacher_b, student_b in heldout_batches:
                teacher_b = teacher_b.to(device)
                student_b = student_b.to(device)
                vec = vector_provider([idx])
                logits, s_gather, keep = _student_pass(student_b, vec)
                s_logits = _gather_completion_logits(logits, student_b,
                                                     gather=s_gather)
                if loss == "hard":
                    targets = student_b.labels[:, 1:][s_gather]
                    val = F.cross_entropy(s_logits.float(), targets)
                else:
                    # AUTHORITATIVE AND UNCACHED: this pass selects the
                    # restored epoch (EarlyStopper.update).
                    t_logits = _uncached_teacher_logits(teacher_b)
                    if keep is not None and not bool(keep.all()):
                        t_logits = t_logits[keep]
                    t_logp = torch.log_softmax(t_logits.float() / temperature,
                                               -1)
                    s_logp = torch.log_softmax(s_logits.float() / temperature,
                                               -1)
                    val = soft_divergence(loss, t_logp, s_logp)
                k = int(s_gather.sum())
                tot += float(val) * k
                n += k
        return tot / max(n, 1)

    # Startup sanity check + the measured no-op floor the end-of-training
    # guard compares against (same examples, object active vs inactive).
    est0 = _probe()
    step0_loss, noop_floor = est0.active, est0.noop
    startup_loss_check(step0_loss, noop_floor, tag=f"distill_{loss}",
                       log_fn=log_fn)

    stopper: EarlyStopper | None = None
    heldout_curve: list[float] = []
    if heldout_batches and early_stop is not None:
        stopper = EarlyStopper(early_stop)
        # Pre-training baseline point (recorded for the curve, never a
        # restore target — see EarlyStopper.restore_best).
        heldout_curve.append(_heldout_loss())
        if log_fn:
            log_fn({f"distill_{loss}/heldout_loss_init": heldout_curve[-1]},
                   0)

    history: list[float] = []
    step = 0
    epochs_run = 0
    stopped_early = False
    for epoch in range(epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        for bi in order:
            idx, teacher_b, student_b = batches[bi]
            teacher_b, student_b = teacher_b.to(device), student_b.to(device)

            vec = vector_provider([idx])
            student_logits, s_gather, keep = _student_pass(student_b, vec)
            s_logits = _gather_completion_logits(student_logits, student_b,
                                                 gather=s_gather)

            if loss == "hard":
                targets = student_b.labels[:, 1:][s_gather]
                step_loss = F.cross_entropy(s_logits.float(), targets)
            else:
                t_logits = _teacher_logits(bi, idx, teacher_b, keep)
                t_logp = torch.log_softmax(t_logits.float() / temperature, -1)
                s_logp = torch.log_softmax(s_logits.float() / temperature, -1)
                step_loss = soft_divergence(loss, t_logp, s_logp)

            opt.zero_grad()
            step_loss.backward()
            opt.step()
            step += 1
            history.append(float(step_loss.detach()))
            if log_fn:
                vec_norm = float(vec.detach().norm())
                log_fn(
                    {
                        f"distill/{loss}_loss": float(step_loss),
                        "distill/vec_norm": vec_norm,
                        "epoch": epoch,
                    },
                    step,
                )
        epochs_run = epoch + 1
        if stopper is not None:
            hl = _heldout_loss()
            heldout_curve.append(hl)
            if log_fn:
                log_fn({f"distill_{loss}/heldout_loss": hl, "epoch": epoch},
                       step)
            if stopper.update(hl, trainable_params):
                stopped_early = True
                break
    if stopper is not None:
        stopper.restore_best(trainable_params)

    # End-of-training probe on the SAME examples as the startup check,
    # with the kept (best-epoch) params — what the floor guard rules on.
    # (The no-op leg is recomputed; the base is frozen, so it is the
    # startup floor's exact tensors and the ratio stays example-paired.)
    est_fin = _probe()
    final_probe_loss = est_fin.active
    if diagnostics is not None:
        diagnostics.update({
            "loss": loss,
            "positions": positions,
            "response_header_len": response_header_len,
            "step0_loss": step0_loss,
            "noop_floor": noop_floor,
            "noop_floor_sd": est0.noop_sd,
            "noop_floor_sem": est0.noop_sem,
            "final_probe_loss": final_probe_loss,
            "final_probe_sd": est_fin.active_sd,
            "final_probe_sem": est_fin.active_sem,
            "floor_ratio": final_probe_loss / max(noop_floor, 1e-6),
            "floor_ratio_sem": est_fin.ratio_sem,
            "probe_n": est_fin.n_examples,
            "probe_n_batches": est_fin.n_batches,
            "probe_selection": "sha256_rank(prompt_without)",
            "probe_selected_sha8": list(est_fin.selected_sha8 or ()),
            "probe_batch_size": est_fin.batch_size,
            # The per-example values themselves (2026-08-24): paired legs in
            # selection order, aligned element-wise with probe_selected_sha8
            # — so two cells probed on the same examples (e.g. an
            # all-position fit vs its response-masked twin) can be compared
            # PAIRED from disk, with the correlated-error sem, instead of
            # re-probing finished objects on a GPU. Both legs come from
            # est_fin (its no-op leg is recomputed on the frozen base, so
            # the pairing is exact). per_family is None on this path --
            # probe_steered_lm passes no family labels — a stated absence.
            **probe_values_record(active_values=est_fin.active_values,
                                  noop_values=est_fin.noop_values,
                                  per_family=est_fin.per_family),
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
            "heldout_curve": heldout_curve,
            "best_epoch": stopper.best_epoch if stopper else None,
            "best_heldout_loss": stopper.best if stopper else None,
        })
    if floor_ratio_max is not None:
        floor_ratio_check(final_probe_loss, noop_floor,
                          ratio_max=float(floor_ratio_max),
                          tag=f"distill_{loss}", log_fn=log_fn)
    return history


def train_distill(
    model,
    tok,
    examples: list[DistillExample],
    layer: int,
    *,
    loss: str = "soft",
    lr: float = 5e-3,
    weight_decay: float = 0.01,
    epochs: int = 3,
    batch_size: int = 4,
    max_len: int = 768,
    temperature: float = 1.0,
    positions: str = "all",
    skip_bos: bool = True,
    response_header_len: int | None = None,
    instruction_id: str = "",
    seed: int = 0,
    log_fn=None,
    heldout_examples: list[DistillExample] | None = None,
    early_stop: EarlyStopSpec | None = None,
    floor_ratio_max: float | None = None,
    n_probe: int | None = DEFAULT_PROBE_N,
    teacher_prefix_cache: bool = False,
) -> SteeringVector:
    """Fixed-vector distillation (the primary 'compile an SI' recipe)."""
    device = next(model.parameters()).device
    vec = nn.Parameter(torch.zeros(hidden_size(model), device=device))
    diagnostics: dict = {}
    history = train_steered_lm(
        model,
        tok,
        examples,
        layer,
        vector_provider=lambda idxs: vec,
        trainable_params=[vec],
        loss=loss,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        temperature=temperature,
        positions=positions,
        skip_bos=skip_bos,
        response_header_len=response_header_len,
        seed=seed,
        log_fn=log_fn,
        heldout_examples=heldout_examples,
        early_stop=early_stop,
        floor_ratio_max=floor_ratio_max,
        diagnostics=diagnostics,
        n_probe=n_probe,
        teacher_prefix_cache=teacher_prefix_cache,
    )
    return SteeringVector(
        vectors={layer: vec.detach().float().cpu()},
        method=f"distill_{loss}",
        model_name=model.config._name_or_path,
        instruction_id=instruction_id,
        metadata={
            "loss": loss,
            "lr": lr,
            "epochs": epochs,
            "n_examples": len(examples),
            "temperature": temperature,
            "positions": positions,
            "skip_bos": skip_bos,
            "response_header_len": response_header_len,
            "final_loss": history[-1] if history else None,
            # The stopping-point record (spec: every arm's stopping point
            # is justified by held-out loss) + the floor-guard numbers.
            "early_stop": {
                k: diagnostics.get(k)
                for k in ("heldout_curve", "best_epoch", "best_heldout_loss",
                          "epochs_run", "stopped_early")
            },
            "floor_check": {
                k: diagnostics.get(k)
                for k in ("step0_loss", "noop_floor", "noop_floor_sem",
                          "final_probe_loss", "final_probe_sem",
                          "floor_ratio", "floor_ratio_sem", "probe_n",
                          "probe_selection", "probe_selected_sha8",
                          # Additive (2026-08-24): the per-example payload
                          # and the probe's chunking, so the vector sidecar
                          # is paired-analysable too (existing keys above
                          # are untouched — readers depend on them).
                          "probe_n_batches", "probe_batch_size",
                          "probe_active_values", "probe_noop_values",
                          "probe_per_family")
            },
        },
    )
