"""From AlphaSteer to KL-constrained steering: the three-stage probe ladder.

Three fitting procedures for a rank-1 gated edit, sharing ONE deployment
interface and one data pipeline:

    h  <-  h + lam * gate(w . h) * r          (per position, every layer in L_s)

    Stage 1  hard null space   constraint: w  |  benign Gram (eigenvalue cutoff)
    Stage 2  soft covariance   penalty:    beta * w' (Sum_b h h') w
    Stage 3  Fisher/KL         penalty:    beta * w' (Sum_b s_b h h') w,
                                           s_b = r' G_b r  (output-KL pullback)

Each stage strictly generalizes the previous one: stage 3 with s == 1 IS
stage 2 (same code path, test T1), and stage 2 with the benign Gram
eigenvalue-thresholded and beta -> inf recovers stage 1 (test T2). All three
share the master ridge problem

    w* = argmin_w  Sum_m (w.h_m - 1)^2 + beta * w' M_b w + alpha ||w||^2   (M)

differing only in M_b. Rationale for rank-1: AlphaSteer's
target matrix R stacks identical copies of r, so its "learnable matrix"
solution factors as Delta* = u r' exactly — the probe w is the only fitted
object and the d x d parameterization is skipped. extraction/alphasteer.py
keeps the paper-faithful d x d arm untouched; parity between
the two stage-1 forms is covered in tests/test_alphakl.py.

SI STEERING ADAPTATION (same mapping as the alphasteer arm):
- "benign"    (edit must vanish) -> SI-PRESENT renderings (system prompt set);
- "malicious" (edit must fire)   -> SI-ABSENT renderings (the eval
  distribution);
- r -> unit-normalized mu(SI-present) - mu(SI-absent), so +r steers toward
  "as if the SI were present". r is NEVER regularized:
  detection (w) gets the benign penalty, actuation (r) keeps its teeth —
  the rank-1 form enforces the asymmetry by construction.

DEPLOYMENT (shared by all stages): per-position rank-1
edit at every steered layer, prompt AND decode positions alike, gate
optionally ReLU-clamped (clamp_gate=True default — the raw linear gate is
unbounded and can go negative, i.e. anti-SI steering on adversarial
projections; run parity tests with clamp=False). This deliberately differs
from the alphasteer arm's prefill-only last-token broadcast: the ladder's
stage-3 benign certificate lives on GENERATION positions, so the edit must
actually apply there. No BOS skip, PSR's rationale: down-weighting outlier
tokens is the fitted gate's job (stage 3's penalty covers BOS positions
explicitly; for stages 1-2 the last-token-only certificate is a documented
defect of those stages, and the measured-KL verification
catches a BOS blowup).

POSITION CONVENTIONS:
- malicious fit term: LAST PROMPT TOKEN in all stages (where the edit must
  trigger);
- benign penalty: last prompt token in stages 1-2; in stage 3, EVERY real
  position of an unsteered greedy rollout (prompt + completion, teacher-
  forced), plus optional raw-text documents (D_b_un) when
  include_unprompted_in_penalty is set.

CALIBRATION: beta is NOT hand-tuned in stage 3 — it is
bisected against a benign KL budget eps (nats/position) predicted on a
HELD-OUT calibration split, then verified with measured per-position
KL(p || p_delta) (2 teacher-forced forwards per example). Predicted and
measured disagreeing by > 2x is warned loudly and recorded (the escape
hatch — bisecting on measured KL directly — is the caller's move). The
per-example predicted KL uses the DEPLOYED gate (clamped when
clamp_gate is set): the certificate must cover the edit that actually
ships; with clamp=False it coincides with the literal formula.
Units: eps is nats/position, so KL_hat per example is the per-position
MEAN (the per-sample sum normalized by T_b), matching the measured
quantity.

STAGE 2 GETS THE SAME DERIVATION BY DEFAULT (2026-08-24). beta is the
ladder's step size — the weight trading the malicious fit against
disturbing the model — and stage 2 used to take it as a REQUIRED manual
number while the calibration machinery sat one stage over: the shipped
beta=1.0 was a crash workaround (the raise) promoted to a hyperparameter,
and the arm diverged under it (the 2026-08-22 wave: alphakl_soft ended
ABOVE the no-op floor on both SIs — wave-record floor_ratio 2.63/3.14,
ratios against the old aim-prefix floors, since re-measured and
substantially moved, so population-scoped: the direction stands, the
magnitudes are not exact). The certificate
is a property of the RESULTING object (its ws, its deployed gate, the
operating lam) and the model — never of which penalty produced the ws —
so stage 2 with beta=None now bisects its own fit (the last-token
unit-weight Gram penalty, unchanged) against the SAME held-out
Fisher-weighted predicted-KL budget stage 3 uses, harvesting only the
CALIBRATION rollouts to do it. An explicit beta still pins the step size
verbatim (bit-identical to the old manual path). Monotonicity caveat,
measured (2026-08-24, tiny-llama + synthetic): beta -> predicted-KL is
provably nonincreasing when the eval form matches the fit penalty
(d KL/d beta = -2 w' M S^-1 M w <= 0) and empirically clean on the
descending limb where a realistic eps crosses; a rank-DEFICIENT fit Gram
(n_fit < d — impossible in production, pool.fit_floor.dims forces
n >= d) adds a nonzero held-out floor, and fp32 ridge gets noisy above
beta ~1e6, exactly as in the shipped stage-3 bisection. calibrate_beta
is safe under both: it only ever returns an iterate it VERIFIED under
budget and raises when even BETA_HI cannot meet eps — non-monotonicity
can cost optimality (a larger beta than the smallest feasible), never an
over-budget certificate; the measured verification below backstops it.

When the HARNESS picks the operating scale (lam None — the canary
base-scale calibration), the canary's choice is additionally gated on that
MEASURED benign KL: the operating point must fit the ladder-wide benign
budget (default BENIGN_KL_BUDGET_FACTOR x eps), stepping DOWN the scale
ladder until it does and REFUSING loudly when even the smallest rung
drifts (see cap_scale_to_benign_budget — the 2026-08-22 wave fix: the
canary's distinct-3/perplexity guard is structurally blind to this arm's
degeneration mode, which is low-perplexity mode collapse, not gibberish).

STATED APPROXIMATIONS (kept and surfaced, not hidden):
second-order KL expansion (checked by the measured pass); diagonal-in-
position penalty with joint-tangent sensitivities; per-layer fits at
Delta=0; empirical-slice certificate (report per slice — D_b_pr says
nothing about D_b_un).

fp32 for Grams, softmaxes, and solves; bf16 activations are fine to
harvest.
"""

from __future__ import annotations

import contextlib
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.func import jvp

from ..generate import encode_kwargs, generate_with_object
from ..hooks import PositionMaskedFit
from ..models import build_chat_prompt, get_decoder_layers, hidden_size
from .calibrate import DEFAULT_SCALE_LADDER, calibrate_base_scale
from .capture import capture_layer_outputs, collect_pooled_activations

SPEC = "AlphaSteer -> KL-constrained steering ladder (user spec 2026-08-18)"

# LadderConfig defaults.
DEFAULT_ALPHA = 1e-2
DEFAULT_TAU = 1e-5          # stage 1 only (the spec's documented hidden knob)
DEFAULT_EPS = 0.01          # nats/position benign budget (stages 2-3)
DEFAULT_ROLLOUT_TOKENS = 128
DEFAULT_MC_SAMPLES = 8
# The benign-KL operating-point gate's default budget, as a multiple of eps:
# measured-KL may legitimately run up to ~2x the second-order prediction
# (the spec's own section-5.4 tolerance band), so a calibrated operating
# point is allowed 2x the eps budget MEASURED before the gate steps the
# scale down / refuses. eps is the LADDER-WIDE benign contract (stage 3
# certifies it ex ante by bisecting beta; stages 1-2 previously enforced
# NOTHING — the 2026-08-22 wave gap), so the gate applies to every stage's
# canary-calibrated scale.
BENIGN_KL_BUDGET_FACTOR = 2.0
# ---- stage-3 capture cap: a MEMORY budget, never a bare position count ----
# Stage 3's harvest captures activations fp32-on-CPU as [T_total, d] PER
# CAPTURED LAYER, where T_total = every real position of every fit rollout.
# Uncapped, the realised two-SI pool is fatal: ~23,000 pairs/SI whose
# 8192-max_len SIs render ~6-7k-token rollout rows -> T_total ~ 1.6e8
# positions ~= 2.6 TB at ONE layer (the 2048-max_len safety arm still needs
# ~150-450 GB) against the fit template's --mem=128G single-GPU allocation
# (scripts/two_si_overnight.py; the arm runs UNSHARDED, so the budget is
# whole-process RAM). Resident bytes = positions x 4 B x d x n_layers, so
# the default cap is derived FROM the byte budget at the model's real d and
# captured-layer count — a fixed position count that fits a one-layer
# capture OOMs a ten-layer one. At d=4096 fp32 (16 KiB per position per
# layer) the 40 GiB budget resolves to:
#   1 captured layer (this lane):    2,621,440 positions (~640 x d)
#   10 captured layers (peer lane):    262,144 positions  (~64 x d)
# Whole-process peak vs --mem=128G: fit capture <= 40 GiB resident,
# + benign_penalty's weighted-Gram transient (s.unsqueeze(1) * Hb, built
#   one layer at a time: <= that layer's slice — 40 GiB at 1 layer, 4 at 10)
# + the calib harvest (same cap; ~8 GB naturally at 64 rows x 1 layer)
# + id rows / randperm scratch / runtime ~ 4 GB
# ~= 92 GB at 1 layer, ~88 GB at 10 — comfortable headroom with the model
# on the GPU. Capture stays fp32 (the downstream solves' convention; bf16
# would halve a number the cap already bounds while perturbing the
# second-moment numerics).
DEFAULT_CAPTURE_BUDGET_BYTES = 40 * 2**30

# The rank floor. Gram rank comes from SAMPLED POSITIONS, and on this
# rollout/full-sequence harvest positions >> prompts (a few hundred rows
# already clear d=4096), so the DATA can never produce a rank deficiency
# here — only a misconfigured cap can. This floor's one job is therefore
# to stop a MEMORY setting from silently becoming a RANK problem:
# train_alphakl RAISES (never warns) when the resolved cap falls below
# this multiple of d. Contrast pool.fit_floor.dims (configs/two_si.yaml),
# which gates a DIFFERENT quantity — matched PAIRS for the LAST-TOKEN arms
# (alphakl_null/_soft, where positions == prompts) — and does not protect
# this path.
MIN_CAPTURE_POSITIONS_PER_DIM = 8


def default_max_capture_positions(
    d: int, n_layers: int,
    budget_bytes: int = DEFAULT_CAPTURE_BUDGET_BYTES,
) -> int:
    """The "auto" capture cap: positions = budget // (4 B x d x n_layers)
    — fp32 bytes per position summed across every captured layer."""
    return max(1, int(budget_bytes) // (4 * int(d) * max(1, int(n_layers))))
BETA_LO, BETA_HI = 1e-4, 1e8


# ---------------------------------------------------------------- fitting ----


def ridge(Hm: torch.Tensor, Mb: torch.Tensor, beta: float,
          alpha: float) -> torch.Tensor:
    """The master problem (M) in closed form.

    Hm: [Nm, d] malicious activations (harness orientation — examples in
    rows, the transpose of the spec's snippet); Mb: [d, d] PSD benign
    penalty. alpha > 0 doubles as the numerical jitter the spec calls for
    when the condition number blows up."""
    Hm = Hm.float()
    d = Hm.shape[1]
    A = Hm.t() @ Hm + beta * Mb.float() + alpha * torch.eye(d, dtype=torch.float32)
    return torch.linalg.solve(A, Hm.sum(dim=0))


def null_basis(
    Hb: torch.Tensor, *, tau: float | None = None,
    nullspace_ratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal basis Q [d, k] of the benign "null space": the bottom
    eigenvectors of the uncentered benign Gram Hb' Hb.

    Two cutoff rules, exactly one required:
    - tau: keep eigenvalues <= tau * max (the spec's stage-1 rule — and its
      named HIDDEN KNOB: the Gram is generically full rank, so tau silently
      decides what counts as benign capability);
    - nullspace_ratio: keep the bottom ratio*d directions (the AlphaSteer
      paper's fraction-of-d rule, matching extraction/alphasteer.py).

    eigh, not svd, for the same reason as alphasteer.null_space_projector
    (f22998f): the empirical Gram is low-rank with ~d - N repeated ~0
    eigenvalues and svd's iterative bidiagonalization fails to converge on
    that spectrum; eigh is the correct solver for a symmetric PSD matrix.
    Returns (Q, eigenvalues ascending)."""
    if (tau is None) == (nullspace_ratio is None):
        raise ValueError("give exactly one of tau / nullspace_ratio")
    A = Hb.float()
    d = A.shape[1]
    gram = A.t() @ A
    try:
        evals, evecs = torch.linalg.eigh(gram)
    except (torch.linalg.LinAlgError, RuntimeError):
        jitter = 1e-6 * float(gram.diagonal().mean().clamp(min=1e-8))
        evals, evecs = torch.linalg.eigh(
            gram + jitter * torch.eye(d, dtype=gram.dtype))
    if nullspace_ratio is not None:
        if not 0.0 < nullspace_ratio < 1.0:
            raise ValueError(f"nullspace_ratio in (0,1), got {nullspace_ratio}")
        k = max(1, int(nullspace_ratio * d))
    else:
        k = max(1, int((evals <= tau * evals[-1]).sum()))
    return evecs[:, :k], evals


def fit_stage1(
    Hm: torch.Tensor, Hb: torch.Tensor, *, alpha: float,
    tau: float | None = None, nullspace_ratio: float | None = None,
) -> torch.Tensor:
    """Stage 1 — AlphaSteer's hard null space, rank-1 form.

    Solve the malicious regression inside the null basis:
    w = Q (Zm' Zm + alpha I)^-1 Zm' 1, Zm = Hm Q. Equivalent to the d x d
    arm's P pinv(X'X + lambda P)(X'R) factored through Delta = u r' (with
    alpha <-> the arm's lambda_reg); the parity test verifies numerically.
    Baseline + parity target; known defects (no exact null space, small-
    variance-high-weight leaks, empirical-last-token-only certificate)
    motivate stages 2-3."""
    Q, _ = null_basis(Hb, tau=tau, nullspace_ratio=nullspace_ratio)
    Zm = Hm.float() @ Q                                    # [Nm, k]
    A = Zm.t() @ Zm + alpha * torch.eye(Q.shape[1], dtype=torch.float32)
    w_hat = torch.linalg.solve(A, Zm.sum(dim=0))
    return Q @ w_hat


def benign_penalty(
    Hb: torch.Tensor, s: torch.Tensor, *, center_gram: bool = False,
    mean_shift_weight: float = 1.0,
) -> torch.Tensor:
    """M_b = Sum_t s_t h_t h_t' (uncentered, the default — penalizing the
    benign mean direction heavily is DESIRABLE as a soft weight; mean
    domination was only a problem for the stage-1 rank cutoff, which it
    eats eigen-directions of).

    center_gram=True (spec's optional LEACE-style flag, off by default):
    s-weighted centered covariance plus an EXPLICIT mean-shift term
    (Sum s) * mu mu' * mean_shift_weight. At weight 1.0 this equals the
    uncentered Gram up to fp error (the cross terms vanish at the weighted
    mean); the weight is the knob that makes the split meaningful."""
    Hb = Hb.float()
    s = s.float()
    if not center_gram:
        return Hb.t() @ (s.unsqueeze(1) * Hb)
    mu = (s.unsqueeze(1) * Hb).sum(0) / s.sum().clamp(min=1e-12)
    Hc = Hb - mu
    return (Hc.t() @ (s.unsqueeze(1) * Hc)
            + mean_shift_weight * s.sum() * torch.outer(mu, mu))


def fit_stage3(
    Hm: torch.Tensor, Hb_pos: torch.Tensor, s_pos: torch.Tensor, *,
    beta: float, alpha: float, center_gram: bool = False,
    mean_shift_weight: float = 1.0,
) -> torch.Tensor:
    """Stage 3 — Fisher/KL-weighted benign penalty.

    Hb_pos: [T, d] benign PER-POSITION activations (rollout harvest);
    s_pos: [T] Fisher weights s_t = r' G_t r. s == 1 reproduces stage 2
    exactly — stage 3 is stage 2 with a metric upgrade."""
    Mb = benign_penalty(Hb_pos, s_pos, center_gram=center_gram,
                        mean_shift_weight=mean_shift_weight)
    return ridge(Hm, Mb, beta, alpha)


def fit_stage2(
    Hm: torch.Tensor, Hb: torch.Tensor, *, beta: float, alpha: float,
    center_gram: bool = False, mean_shift_weight: float = 1.0,
) -> torch.Tensor:
    """Stage 2 — soft covariance penalty: delete the SVD and the cutoff,
    keep the Gram as a continuous penalty. IMPLEMENTED AS stage 3 with
    unit weights — literally the same code path, so test T1's
    bitwise-equality requirement holds by construction."""
    return fit_stage3(Hm, Hb, torch.ones(Hb.shape[0]), beta=beta, alpha=alpha,
                      center_gram=center_gram,
                      mean_shift_weight=mean_shift_weight)


# ------------------------------------------------------- Fisher weights ----


def _sdpa_math_ctx():
    """Force the math SDPA backend during functional-AD passes: the flash /
    mem-efficient kernels do not support forward-mode AD; plain matmul +
    softmax does. No-op fallback where the context manager is missing."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.MATH])
    except ImportError:                                    # pragma: no cover
        return contextlib.nullcontext()


def _write_hooks(model, writes: dict[int, torch.Tensor], eps, steer):
    """Register forward hooks adding (eps * steer)[:, :, None] * r_l to the
    block-l OUTPUT (the harness layer convention) for every steered layer.
    eps may be a forward-AD dual scalar or a [B, T] leaf tensor — the write
    is cast to the hidden dtype AFTER the multiply so the tangent/grad
    survives in fp32 as long as possible. Returns the handles."""
    decoder = get_decoder_layers(model)
    handles = []

    def make(r):
        def hook(_mod, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            add = ((eps * steer).unsqueeze(-1)
                   * r.to(device=hidden.device, dtype=torch.float32))
            steered = hidden + add.to(hidden.dtype)
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        return hook

    for l, r in writes.items():
        handles.append(decoder[l].register_forward_hook(make(r)))
    return handles


def fisher_weights_jvp(
    model, writes: dict[int, torch.Tensor], input_ids: torch.Tensor,
    attention_mask: torch.Tensor, steer_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """s_t = Var_{y ~ p_t}[u_t[y]] per position, u = J . (joint unit write):
    the pullback of output KL to the steered activation space, evaluated
    along the write direction(s), via ONE forward-mode JVP per batch.

    The tangent is taken w.r.t. a SCALAR epsilon multiplying the unit write
    of r_l at every steered layer and position simultaneously — the joint
    multi-layer, all-positions tangent (the conservative default):
    d logits / d eps at eps=0 IS J . (joint write).
    Tangents only propagate from the injection sites, so the marginal cost
    is ~the layers above min(L_s). Runs under no_grad (forward-mode needs
    no reverse graph — verified) with math-backend SDPA. Returns s [B, T]
    fp32, zeroed outside steer_mask."""
    device = input_ids.device
    steer = (attention_mask if steer_mask is None else steer_mask)
    steer = steer.to(device=device, dtype=torch.float32)

    def f(eps):
        handles = _write_hooks(model, writes, eps, steer)
        try:
            return model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=False).logits
        finally:
            for h in handles:
                h.remove()

    with torch.no_grad(), _sdpa_math_ctx():
        logits, dlogits = jvp(
            f, (torch.zeros((), device=device),),
            (torch.ones((), device=device),))
    # Position-chunked fp32 statistics: a full-batch .float().softmax(-1)
    # materializes TWO [B, T, V] fp32 tensors — ~25 GB apiece at B=8,
    # T~6k, V~128k (a long-system-prompt SI). Chunking over T keeps the
    # fp32 peak at chunk x V while the math is unchanged.
    B, T, _ = logits.shape
    out = torch.empty(B, T, dtype=torch.float32)
    for i in range(0, T, 2048):
        p = logits[:, i:i + 2048].float().softmax(-1)
        u = dlogits[:, i:i + 2048].float()
        m1 = (p * u).sum(-1)
        m2 = (p * u * u).sum(-1)
        out[:, i:i + 2048] = (m2 - m1 * m1).cpu()
    return out * steer.cpu()


def fisher_weights_mc(
    model, writes: dict[int, torch.Tensor], input_ids: torch.Tensor,
    attention_mask: torch.Tensor, steer_mask: torch.Tensor | None = None,
    *, mc_samples: int = DEFAULT_MC_SAMPLES, seed: int = 0,
) -> torch.Tensor:
    """MC reverse-mode alternative (fisher_mode="mc_vjp").

    Per-position epsilons scale the joint write at each position
    independently; y_t ~ p_t sampled independently (teacher-forced, NOT
    autoregressive), score = d/d eps_t Sum_t' log p(y_t'), and
    s_hat_t = mean_K[score_t^2] — the score has mean 0 and covariance =
    Fisher. Estimator semantics differ from the JVP mode (this attributes
    to position t the sensitivity of all DOWNSTREAM positions to a write
    at t; the JVP mode measures position t's own sensitivity to the joint
    write) — both are diagonal summaries of the same PSD form, and s is a
    regression weight averaged over many positions, so the noise and the
    semantic gap are acceptable (spec's words). Cost ~2K forwards
    independent of |L_s|."""
    device = input_ids.device
    steer = (attention_mask if steer_mask is None else steer_mask)
    steer = steer.to(device=device, dtype=torch.float32)
    eps = torch.zeros(input_ids.shape, device=device, requires_grad=True)

    handles = _write_hooks(model, writes, eps, steer)
    try:
        with _sdpa_math_ctx(), torch.enable_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           use_cache=False).logits
            logp = logits.float().log_softmax(-1)
            B, T, V = logp.shape
            gen = torch.Generator(device="cpu").manual_seed(seed)
            probs = logp.detach().exp().reshape(B * T, V).cpu()
            acc = torch.zeros(B, T)
            for k in range(mc_samples):
                y = torch.multinomial(probs, 1, generator=gen)
                y = y.reshape(B, T).to(device)
                loss = (logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                        * steer).sum()
                (g,) = torch.autograd.grad(
                    loss, eps, retain_graph=(k < mc_samples - 1))
                acc += (g.float() ** 2).cpu()
    finally:
        for h in handles:
            h.remove()
    return (acc / mc_samples) * steer.cpu()


# ------------------------------------------------- rollouts and harvests ----


@torch.no_grad()
def rollout_ids(
    model, tok, rendered: list[str], *, max_new_tokens: int = 128,
    batch_size: int = 8, do_sample: bool = False,
) -> list[torch.Tensor]:
    """Unsteered greedy rollout (the stage-3 position source), returned as
    EXACT token ids (prompt + completion, per example, pads stripped,
    truncated after the first EOS) so the teacher-forced harvest and the
    measured-KL passes replay precisely what the model generated — no
    decode/re-tokenize drift."""
    device = next(model.parameters()).device
    prev_side = tok.padding_side
    tok.padding_side = "left"
    rows: list[torch.Tensor] = []
    try:
        for i in range(0, len(rendered), batch_size):
            chunk = rendered[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      **encode_kwargs(tok, chunk[0])).to(device)
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                pad_token_id=tok.pad_token_id)
            width = enc["input_ids"].shape[1]
            for b in range(out.shape[0]):
                prompt = out[b, :width][enc["attention_mask"][b].bool()]
                gen = out[b, width:]
                if tok.eos_token_id is not None:
                    stop = (gen == tok.eos_token_id).nonzero()
                    if stop.numel():
                        gen = gen[: int(stop[0, 0]) + 1]
                rows.append(torch.cat([prompt, gen]).cpu())
    finally:
        tok.padding_side = prev_side
    return rows


def text_ids(tok, texts: list[str]) -> list[torch.Tensor]:
    """Tokenize raw documents (the D_b_un slice: plain continuation, no chat
    template) into harvest-ready id rows."""
    return [torch.tensor(tok(t)["input_ids"]) for t in texts]


def _pad_rows(rows: list[torch.Tensor], pad_id: int, device):
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for b, r in enumerate(rows):
        ids[b, : len(r)] = r
        mask[b, : len(r)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def _forward_capture(model, ids, mask, layers):
    with capture_layer_outputs(model, layers) as captured:
        model(input_ids=ids, attention_mask=mask, use_cache=False)
    return captured


def harvest_positions(
    model, id_rows: list[torch.Tensor], layers: list[int], *, pad_id: int,
    writes: dict[int, torch.Tensor] | None = None, batch_size: int = 8,
    fisher_mode: str = "jvp", mc_samples: int = DEFAULT_MC_SAMPLES,
    seed: int = 0, max_positions: int | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor | None, torch.Tensor]:
    """Teacher-force id_rows, harvesting EVERY real position: per-layer
    activations (fp32 cpu, [T_total, d]) and, when `writes` is given, the
    per-position Fisher weights s (one extra JVP or MC pass per chunk).
    Returns (acts, s | None, example_index [T_total]).

    ``max_positions`` caps the CAPTURE (the RAM knob above): when T_total
    exceeds it, a seed-deterministic uniform sample of ``max_positions``
    POSITIONS is drawn across ALL rows jointly — deliberately never a
    row/pair cap. Every row keeps ~len/T_total of the sample, so prompt
    diversity survives; and because the benign Gram and the fisher-weight
    (JVP/MC) kernel are POSITION-AVERAGED estimators — every position
    weighted equally — a uniform sample over the joint position axis
    estimates both unbiasedly (up to scale), whereas a row/pair cap
    concentrates the estimate in whichever prompts survive it. The
    selection depends only on (row lengths, seed,
    max_positions) — the identical rule for every SI — and every forward
    still runs on every row: the cap bounds resident CPU RAM, not compute
    or which pairs contribute."""
    if fisher_mode not in ("jvp", "mc_vjp"):
        raise ValueError(f"fisher_mode must be jvp|mc_vjp, got {fisher_mode!r}")
    if max_positions is not None and int(max_positions) < 1:
        raise ValueError(f"max_positions must be >= 1, got {max_positions!r}")
    device = next(model.parameters()).device
    total = sum(len(r) for r in id_rows)
    sel: torch.Tensor | None = None
    if max_positions is not None and total > int(max_positions):
        g = torch.Generator().manual_seed(int(seed))
        sel = torch.zeros(total, dtype=torch.bool)
        sel[torch.randperm(total, generator=g)[: int(max_positions)]] = True
    acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    weights: list[torch.Tensor] = []
    ex_idx: list[torch.Tensor] = []
    offset = 0
    for i in range(0, len(id_rows), batch_size):
        rows = id_rows[i:i + batch_size]
        ids, mask = _pad_rows(rows, pad_id, device)
        keep = mask.bool()
        bsel: torch.Tensor | None = None
        if sel is not None:
            # keep-flattening is row-major (right padding), so the global
            # position order is exactly rows-in-order / tokens-in-order —
            # slice this batch's segment of the selection mask.
            n_batch = sum(len(r) for r in rows)
            bsel = sel[offset:offset + n_batch]
            offset += n_batch
        captured = _forward_capture(model, ids, mask, layers)
        for l in layers:
            flat = captured[l][keep]
            if bsel is not None:  # subset BEFORE the fp32 CPU copy
                flat = flat[bsel.to(flat.device)]
            acts[l].append(flat.float().cpu())
        if writes is not None:
            fw = (fisher_weights_jvp if fisher_mode == "jvp"
                  else fisher_weights_mc)
            kw = {} if fisher_mode == "jvp" else {
                "mc_samples": mc_samples, "seed": seed + i}
            s = fw(model, writes, ids, mask, **kw)
            s_kept = s[keep.cpu()]
            weights.append(s_kept if bsel is None else s_kept[bsel])
        bidx = torch.cat([torch.full((len(r),), i + b, dtype=torch.long)
                          for b, r in enumerate(rows)])
        ex_idx.append(bidx if bsel is None else bidx[bsel])
    return (
        {l: torch.cat(v) for l, v in acts.items()},
        torch.cat(weights) if writes is not None else None,
        torch.cat(ex_idx),
    )


# ------------------------------------------------ predicted / measured KL ----


def _gates(ws: dict[int, torch.Tensor], acts: dict[int, torch.Tensor],
           clamp: bool) -> torch.Tensor:
    """Mean-over-layers deployed gate per position, [T]. Exact for a single
    steered layer; for several it treats the per-layer writes as parallel in
    function space — consistent with the shared-s joint-tangent
    approximation, and the measured pass is the check.

    The probe is aligned to the ACTIVATIONS' device: harvests live on CPU
    while a deployed object's ws sit on CUDA after obj.to_ — the 147484
    canary (all 3 rungs) died exactly here at the post-fit diagnostics,
    invisible to the CPU-only test fixture."""
    g = torch.stack([
        acts[l].float() @ ws[l].float().to(acts[l].device) for l in ws
    ]).mean(0)
    return g.clamp_min(0.0) if clamp else g


def predicted_benign_kl(
    ws: dict[int, torch.Tensor], lam: float, acts: dict[int, torch.Tensor],
    s: torch.Tensor, ex_idx: torch.Tensor, *, clamp: bool = False,
) -> dict:
    """KL_hat: per example, mean over its positions of
    (1/2) lam^2 gate_t^2 s_t — nats/position, so it compares directly to
    the measured quantity and to eps. clamp=True uses the DEPLOYED
    (ReLU'd) gate; False is the spec's literal (conservative) formula."""
    g = _gates(ws, acts, clamp)
    per_pos = 0.5 * lam * lam * g * g * s.float()
    n_ex = int(ex_idx.max()) + 1
    sums = torch.zeros(n_ex).index_add_(0, ex_idx, per_pos)
    counts = torch.zeros(n_ex).index_add_(
        0, ex_idx, torch.ones_like(per_pos))
    per_example = sums / counts.clamp(min=1)
    return {"mean": float(per_example.mean()),
            "per_example": per_example.tolist()}


@torch.no_grad()
def measured_benign_kl(
    model, obj, id_rows: list[torch.Tensor], *, pad_id: int,
    scale: float = 1.0, batch_size: int = 8,
) -> dict:
    """Exact per-position KL(p || p_delta) between the unsteered and steered
    next-token distributions, teacher-forced on the SAME ids (the unsteered
    rollouts) — 2 forwards per example (mandatory verification; also the
    harness's primary benign-drift metric, valid regardless of how w was
    fit). Returns nats/position, mean-of-example-means."""
    device = next(model.parameters()).device
    per_example: list[float] = []
    for i in range(0, len(id_rows), batch_size):
        rows = id_rows[i:i + batch_size]
        ids, mask = _pad_rows(rows, pad_id, device)
        base = model(input_ids=ids, attention_mask=mask,
                     use_cache=False).logits
        with obj.apply(model, scale=scale):
            steered = model(input_ids=ids, attention_mask=mask,
                            use_cache=False).logits
        logp = base.float().log_softmax(-1)
        logq = steered.float().log_softmax(-1)
        kl = (logp.exp() * (logp - logq)).sum(-1)          # [B, T]
        m = mask.bool()
        for b in range(len(rows)):
            per_example.append(float(kl[b][m[b]].mean()))
    return {"mean": float(torch.tensor(per_example).mean()),
            "per_example": per_example}


# ---------------------------------------------------------- calibration ----


def calibrate_beta(
    fit_fn, kl_fn, eps: float, *, lo: float = BETA_LO, hi: float = BETA_HI,
    iters: int = 40,
) -> tuple[float, dict[int, torch.Tensor]]:
    """Bisect beta (log-space) against the held-out predicted budget:
    fit_fn(beta) -> ws; kl_fn(ws) -> nats/position.
    Returns the smallest tested beta whose fit is UNDER budget (with its
    ws) — never an over-budget iterate. Fails loudly when even beta=hi
    can't meet eps (an unmeetable budget must not degrade silently)."""
    best: tuple[float, dict] | None = None
    for _ in range(iters):
        beta = math.sqrt(lo * hi)
        ws = fit_fn(beta)
        if kl_fn(ws) > eps:
            lo = beta
        else:
            hi = beta
            best = (beta, ws)
    if best is None:
        ws = fit_fn(hi)
        if kl_fn(ws) > eps:
            raise RuntimeError(
                f"calibrate_beta: even beta={hi:g} cannot meet "
                f"eps={eps:g} nats/position — the budget is unmeetable "
                "on this data (raise eps or inspect the harvest)")
        best = (hi, ws)
    return best


def cap_scale_to_benign_budget(
    measure_fn, *, base_scale: float, scale_ladder, budget: float,
    arm: str = "arm",
) -> tuple[float, list[dict], bool]:
    """Gate the canary-calibrated operating point on MEASURED benign KL:
    walk down from ``base_scale`` through the smaller ladder rungs until
    ``measure_fn(scale)`` (mean nats/position of exact KL(p || p_delta) on
    the held-out SI-PRESENT rollouts) fits ``budget``; REFUSE loudly when
    even the smallest rung drifts. Returns (scale, trail, capped) — trail
    is the audit record of every (scale, measured KL) tried, capped says
    the canary's own choice was rejected.

    WHY (the 2026-08-22 wave, safety_refusal/si_constitution alphakl_soft):
    the shared canary guard (calibrate_base_scale) flags distinct3 < 0.35
    or gen_ppl > 3x base — the signature of over-steering into GIBBERISH,
    which is how the alphasteer arm degenerates (its wave refusal recorded
    gen_ppl 84.8 at the smallest rung). This arm's rank-1 ReLU-gated
    every-position edit degenerates the OTHER way, into low-perplexity
    MODE COLLAPSE: on safety_refusal the steered gen_ppl fell 10.52 (base)
    -> 5.26 (scale 8) -> 4.31 (16) -> 3.72 (32) while distinct3 held
    0.55-0.62 until verbatim looping at 32 (0.066). A one-sided ppl guard
    cannot fire on FALLING perplexity, so the ladder mislabelled deeply
    collapsed rungs "coherent", hi landed at 16, and the midpoint rule
    shipped base_scale=8 — an operating point whose measured benign drift
    was 0.719 nats/position (72x the eps=0.01 ladder budget) on the very
    rollouts where this arm's edit is contractually supposed to VANISH.
    si_constitution shipped 4 at 0.0627 the same way. Both objects moved
    the student AWAY from the teacher (floor_ratio 2.63 / 3.14 vs a
    no-op's 1.0); the floor guard caught only one of them, by luck of a
    lower floor. The trainer already computed the measured number both
    times — this gate makes it binding instead of decorative.

    The refusal mirrors calibrate_base_scale's "NO coherent scale": a
    method that cannot operate within its own benign budget at ANY scale
    must say so, not ship the least-degenerate large scale. A capped scale
    is only reduced as far as the budget requires (never silently to ~0:
    the floor is the smallest ladder rung, below which we refuse)."""
    cands = [float(base_scale)] + sorted(
        {float(s) for s in scale_ladder if float(s) < float(base_scale)},
        reverse=True)
    trail: list[dict] = []
    for cand in cands:
        kl = float(measure_fn(cand))
        trail.append({"scale": cand, "kl_measured_prompted": kl})
        if kl <= budget:
            return cand, trail, cand != cands[0]
    raise RuntimeError(
        f"{arm} calibration: NO scale meets the benign KL budget "
        f"({budget:g} nats/position, measured on held-out SI-PRESENT "
        f"rollouts, where this arm's edit must vanish). The canary picked "
        f"base_scale={cands[0]:g} ({trail[0]['kl_measured_prompted']:.4g} "
        "nats/position) and every smaller ladder rung still exceeds the "
        f"budget. Trail: {trail}. The canary's distinct-3/perplexity guard "
        "cannot catch this failure mode (over-steering here is fluent "
        "low-perplexity mode collapse, not gibberish), so the measured-KL "
        "gate is the binding check. Refusing to ship a worse-than-no-op "
        "object; pass lam to pin a scale deliberately, or raise "
        "benign_kl_budget.")


def _rank_auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUC of pos-vs-neg scores (no sklearn dependency)."""
    scores = torch.cat([pos, neg])
    ranks = scores.argsort().argsort().float() + 1.0
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float("nan")
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


# ------------------------------------------------------------- the object ----


class AlphaKLObject(PositionMaskedFit):
    """The ladder's shared deployment: per steered layer
    l, h <- h + base_scale*scale * gate(w_l . h) * r_l at EVERY position,
    prefill and decode alike; gate = ReLU when clamp_gate (blocks
    anti-SI steering on negative projections), raw linear otherwise
    (paper-parity mode). Inference overhead ~2d FLOPs/token/layer,
    identical across stages — the stages differ only in how w was fit.

    SERVE-side position masking ONLY: the mixin's
    ``set_fit_positions`` seam is consumed here exclusively by the serving
    surface (``jailbreak._set_response_positions``) so the arm's decode-only
    cell runs under the POSITION MASK — the header + generated positions,
    steering the forward that produces the FIRST generated token — never the
    engine-phase gate (unsteered prefill, steering from token 2), whose
    token-1-unsteered numbers are not comparable. The edit is per-position
    already (add = mag * gate(w.h) * r over [B, S]), so masking is exact:
    masked-out positions stay BIT-identical via ``_fit_masked``.

    The FIT never masks: ``train_alphakl`` REFUSES ``positions != "all"``
    (a masked fit would solve for the same (w, r) and merely relabel the
    artifact — see the raise there), the arm stays out of ``class_c_arms``,
    ``save()`` deliberately persists no position leg, and every alphakl
    artifact's ``positions_at_fit`` records the all-position fit."""

    kind = "alphakl"

    def __init__(self, model, layers: list[int], ws: dict[int, torch.Tensor],
                 rs: dict[int, torch.Tensor], base_scale: float = 1.0,
                 clamp_gate: bool = True, stage: int | None = None,
                 metadata: dict | None = None):
        self.layers = list(layers)
        self.ws = {int(l): ws[l].detach().float() for l in self.layers}
        self.rs = {int(l): rs[l].detach().float() for l in self.layers}
        self.base_scale = float(base_scale)
        self.clamp_gate = bool(clamp_gate)
        self.stage = stage
        self.metadata = metadata or {}
        self._decoder = get_decoder_layers(model)

    @property
    def trainable(self) -> list:
        return []  # closed-form; nothing trains by gradient

    def n_params(self) -> int:
        return sum(w.numel() for w in self.ws.values())

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def gate(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """gate(w . h) per token: [B, S], fp32."""
        g = hidden.float() @ self.ws[layer].to(hidden.device)
        return g.clamp_min(0.0) if self.clamp_gate else g

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mag = self.base_scale * scale
        handles = []
        # Serve-side mask tracker (None on the default all-position leg —
        # that path is byte-identical to the pre-mask behavior). The mask
        # comes from SteeringHandle._position_mask, the single place a mask
        # is computed; masked-out positions stay bit-identical (torch.where).
        mask_handle = self._fit_mask_handle(model)

        def make(layer):
            def hook(_mod, _args, output):
                if mag == 0.0:
                    return output  # exact no-op, bit-identical to unsteered
                hidden = output[0] if isinstance(output, tuple) else output
                g = self.gate(layer, hidden)               # [B, S]
                r = self.rs[layer].to(hidden.device)
                add = mag * g.unsqueeze(-1) * r            # fp32
                steered = hidden + add.to(hidden.dtype)
                steered = self._fit_masked(mask_handle, steered, hidden)
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered
            return hook

        for l in self.layers:
            handles.append(self._decoder[l].register_forward_hook(make(l)))
        try:
            yield mask_handle
        finally:
            for h in handles:
                h.remove()
            if mask_handle is not None:
                mask_handle.remove()

    def to_(self, device) -> None:
        self.ws = {l: w.to(device) for l, w in self.ws.items()}
        self.rs = {l: r.to(device) for l, r in self.rs.items()}

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind, "layers": self.layers,
             "ws": {l: w.detach().float().cpu() for l, w in self.ws.items()},
             "rs": {l: r.detach().float().cpu() for l, r in self.rs.items()},
             "base_scale": self.base_scale, "clamp_gate": self.clamp_gate,
             "stage": self.stage, "metadata": self.metadata},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "AlphaKLObject":
        return cls(
            model, layers=[int(l) for l in blob["layers"]],
            ws={int(l): w for l, w in blob["ws"].items()},
            rs={int(l): r for l, r in blob["rs"].items()},
            base_scale=float(blob["base_scale"]),
            clamp_gate=bool(blob.get("clamp_gate", True)),
            stage=blob.get("stage"),
            metadata=blob.get("metadata", {}),
        )


# ------------------------------------------------------- registry trainer ----


def _subsample_record(rows: list[torch.Tensor], idx: torch.Tensor,
                      sub_seed: int) -> dict:
    """One harvest leg's capture-cap provenance (recorded into obj.metadata
    and lifted into the fit report + train_budget_meta.json by the fit
    driver): realised positions, realised contributing rows, sampling seed
    — a capped harvest must never look like an uncapped one. "pairs" =
    harvest rows (each pair's rollout, plus any unprompted docs when
    include_unprompted_in_penalty)."""
    tot = sum(len(r) for r in rows)
    return {"seed": int(sub_seed),
            "total_positions": int(tot),
            "kept_positions": int(idx.numel()),
            "capped": bool(idx.numel() < tot),
            "pairs_total": len(rows),
            "pairs_contributing": int(idx.unique().numel())}


def train_alphakl(
    model,
    tok,
    instruction,
    pairs,
    *,
    layer: int,
    stage: int = 3,
    layers: list[int] | None = None,
    log_fn=None,
    alpha: float = DEFAULT_ALPHA,
    tau: float = DEFAULT_TAU,
    nullspace_ratio: float | None = None,
    beta: float | None = None,
    eps: float = DEFAULT_EPS,
    lam: float | None = None,
    clamp_gate: bool = True,
    center_gram: bool = False,
    mean_shift_weight: float = 1.0,
    fisher_mode: str = "jvp",
    mc_samples: int = DEFAULT_MC_SAMPLES,
    max_capture_positions: int | str | None = "auto",
    rollout_tokens: int = DEFAULT_ROLLOUT_TOKENS,
    unprompted_texts: list[str] | None = None,
    include_unprompted_in_penalty: bool = False,
    calib_frac: float = 0.25,
    max_calib: int = 64,
    batch_size: int = 8,
    calib_max_new_tokens: int = 128,
    scale_ladder=DEFAULT_SCALE_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    benign_kl_budget: float | None = None,
    beta_iters: int = 40,
    seed: int = 0,
    epochs: int | None = None,   # train_kwargs.default compat (documented no-op)
    max_len: int | None = None,  # train_kwargs.default compat (documented no-op)
    positions: str = "all",
    response_header_len: int | None = None,
    # Teacher-side prefix KV caching for the four
    # last-token capture sweeps below — the SI-present renderings share the
    # chat header + full SI as a literal token prefix, so the flag runs it
    # through the model once per sweep instead of once per prompt
    # (capture.py's cached path; semantically exact, fp-tolerance-only).
    # The stage-3 rollout/Fisher forwards are NOT prefix-shaped teacher
    # legs and are untouched. Default False keeps other lanes
    # byte-identical; the two-SI driver passes the config flag.
    teacher_prefix_cache: bool = False,
) -> AlphaKLObject:
    """Registry trainer for the ladder (any stage).

    Splits `pairs` into FIT and held-out CALIBRATION tails (calibration
    disjoint from fit — an honest-split departure from the sibling
    closed-form arms, which canary on in-sample prompts), captures
    last-token activations, computes the unit refusal direction, fits the
    stage's probe, then:

    beta (stages 2-3): None — the DEFAULT — derives it by bisecting the
    stage's fit against the eps benign budget, predicted on the held-out
    calibration harvest (calibrate_beta; stage 2 gained this on 2026-08-24
    — see the module docstring's stage-2 calibration paragraph for why and
    for the measured monotonicity caveats). Stage 2's FIT penalty stays the
    last-token unit-weight Gram either way; deriving only adds the
    calibration-split rollout + Fisher harvest that prices the certificate
    in nats/position. An explicit beta pins the step size verbatim
    (stage-2 pinned fits are bit-identical to the pre-derivation path) —
    for reproducing a recorded run, never the default. The derived value
    is logged and lands in metadata ("beta", "beta_derived_from_eps"), so
    a reader can always tell a derived step size from a pinned one.

    - lam given -> base_scale = lam VERBATIM (the KL budget certificate is
      issued AT lam; the eval factor grid sweeps around it);
    - lam None  -> canary base-scale calibration (the harness's operating-
      point rule, standing in for the spec's refusal-rate target, which has
      no cheap analogue at fit time) on the CALIB malicious renderings. In
      stages 2-3 with a derived beta, beta is then RE-BISECTED at the
      calibrated scale and the probe refit — one fixed-point pass, so the
      budget is certified at the scale that actually deploys (the bisection
      is seconds of d x d solves; the canary is not repeated, so a large
      w change would leave the scale slightly stale — recorded, and the
      measured verification is the backstop). The canary's choice is then
      GATED on measured benign KL (``benign_kl_budget``, default
      BENIGN_KL_BUDGET_FACTOR x eps nats/position): the canary's
      distinct-3/ppl guard cannot see this arm's low-perplexity mode
      collapse (2026-08-22 wave: base_scale=8 shipped at 0.719
      nats/position of benign drift), so the operating scale is stepped
      down the ladder until the measured drift fits the budget, and the
      fit REFUSES (RuntimeError, mirroring the canary's own "NO coherent
      scale") when even the smallest rung drifts — see
      cap_scale_to_benign_budget. The trail is recorded in
      ``metadata["benign_kl_gate"]`` and logged per step. A capped
      derived-beta scale keeps the beta bisected at the canary scale
      (predicted KL is monotone in lam, so the eps certificate still
      covers the smaller deployed lam — conservatively). A GIVEN lam
      bypasses the gate entirely: the spec pins user-supplied lam verbatim.

    ``max_capture_positions`` (the ``train_kwargs.alphakl_fisher`` knob)
    bounds the harvests' resident activations (stage 3's fit + calib legs;
    the stage-2 derived-beta calib leg): the default "auto" derives the
    cap from ``DEFAULT_CAPTURE_BUDGET_BYTES`` at the model's hidden size
    and len(layers) (module constants carry the arithmetic), an int is
    used verbatim, and None runs uncapped. The resolved cap, the realised
    positions/rows and the sampling seed are recorded in
    ``metadata["position_subsample"]``.

    Always finishes with the MANDATORY measured-KL verification on the
    calib rollouts (plus the D_b_un slice when `unprompted_texts` is given
    — the spec's held-out leakage probe), warning loudly when measured and
    predicted disagree by > 2x. epochs/max_len are
    documented no-ops (closed-form extraction).

    ``positions`` other than "all" is REFUSED — see the raise below for
    the full reason. The kwarg is accepted (rather than left to a
    TypeError) so the refusal states WHY, not just that the signature
    disagrees."""
    del epochs, max_len
    if positions != "all":
        # A masked (decode-only) fit request. The ladder's fit is a
        # closed-form ridge/probe solve over UNSTEERED activations at
        # pinned positions (malicious term at the LAST PROMPT TOKEN in
        # all stages; stage-3 benign penalty and eps-certificate over
        # EVERY real rollout position) — there is no steered training
        # forward for a position mask to act on, so "fitting decode-only"
        # would solve for the SAME (w, r) and merely relabel the
        # artifact: exactly the silently-pretending-to-be-masked fit the
        # mask→loss wiring exists to prevent. A deployment-matched
        # decode-only ladder needs those position conventions re-derived
        # (restrict the stage-3 penalty/certificate to boundary-onward
        # positions and re-bisect beta) — a method-design change, not a
        # mask-threading one. The test-time decode-only leg is a
        # SERVE-phase choice instead: the all-position artifact is served
        # under the POSITION MASK via jailbreak._set_response_positions
        # (AlphaKLObject's serve-side mask): a decode-only serve must
        # steer the FIRST generated token, so the engine-phase gate,
        # whose prefill runs unsteered and token 1 with it, is not a
        # valid deployment for this leg.
        raise ValueError(
            f"alphakl (stage {stage}) cannot honor a positions="
            f"{positions!r} (decode-only) FIT: the fit is a closed-form "
            "solve on unsteered activations at spec-pinned positions "
            "(malicious term at the last prompt token; stage-3 benign "
            "penalty/eps-certificate over all rollout positions) — no "
            "steered training forward exists for a mask to act on, so a "
            "'masked' fit would produce a bit-identical object under a "
            "decode-only label, and the stage-3 KL certificate would no "
            "longer describe the deployment. Re-derive the ladder's "
            "position conventions (a spec change) before fitting this "
            "leg; the test-time decode-only leg is a SERVE-phase choice "
            "(--positions response — the position-mask serving of the "
            "all-position artifact) and needs no fit change.")
    del response_header_len
    if stage not in (1, 2, 3):
        raise ValueError(f"stage must be 1|2|3, got {stage}")
    # (stage 2 with beta None used to RAISE here — "stage 2 takes a MANUAL
    # beta". That raise is what made the config's beta pin exist at all,
    # and the pinned arm diverged; stage 2 now derives beta from eps by
    # default, exactly as stage 3 does — see the stage-2 fit branch below.)
    prompts = [p.prompt for p in pairs]
    if not prompts:
        raise ValueError("alphakl extraction got an empty pairs list")
    steer_layers = [int(l) for l in (layers or [layer])]

    resolved_cap: int | None = None
    cap_source: str | None = None
    # Stage 3 harvests fit + calib rollouts; stage 2 with beta None (the
    # derived default) harvests the CALIB rollouts only, as its
    # certificate basis. Both are bounded by the same capture cap.
    if stage == 3 or (stage == 2 and beta is None):
        d_model = hidden_size(model)
        if max_capture_positions == "auto":
            resolved_cap = default_max_capture_positions(
                d_model, len(steer_layers))
            cap_source = "auto"
        elif max_capture_positions is None:
            cap_source = "uncapped"
        elif isinstance(max_capture_positions, str):
            raise ValueError(
                "max_capture_positions must be 'auto', an int, or null — "
                f"got {max_capture_positions!r}")
        else:
            resolved_cap = int(max_capture_positions)
            cap_source = "configured"
        # The rank floor guards the stage-3 FIT Gram, which is built from
        # the capped harvest. Stage 2's fit Gram is last-token (cap-free);
        # there the cap only subsamples the CERTIFICATE estimator — still
        # unbiased under the uniform position rule, and backstopped by the
        # mandatory measured verification — so the floor does not apply.
        if resolved_cap is not None and stage == 3:
            floor = MIN_CAPTURE_POSITIONS_PER_DIM * d_model
            if resolved_cap < floor:
                raise ValueError(
                    f"max_capture_positions ({cap_source}) resolves to "
                    f"{resolved_cap}, below the rank floor {floor} "
                    f"(= {MIN_CAPTURE_POSITIONS_PER_DIM} x d={d_model}). "
                    "On this rollout harvest positions >> prompts, so the "
                    "DATA cannot produce a rank-deficient Gram — only this "
                    "memory knob can, and a memory setting must never "
                    "silently become a rank problem. (pool.fit_floor.dims "
                    "gates a DIFFERENT quantity: matched PAIRS for the "
                    "last-token arms, where positions == prompts; it does "
                    "not protect this path.) Raise the cap or the capture "
                    "budget, or set the knob null for the uncapped "
                    "harvest.")

    n_calib = min(max_calib, max(2, round(calib_frac * len(prompts))))
    if len(prompts) - n_calib < 2:
        raise ValueError(
            f"{len(prompts)} pairs cannot fund a fit + {n_calib}-prompt "
            "calibration split — give more pairs or lower calib_frac")
    fit_prompts, calib_prompts_l = prompts[:-n_calib], prompts[-n_calib:]

    def render(ps, system):
        return [build_chat_prompt(tok, p, system=system) for p in ps]

    benign_fit = render(fit_prompts, instruction.text)     # SI present
    malicious_fit = render(fit_prompts, None)              # SI absent
    benign_cal = render(calib_prompts_l, instruction.text)
    malicious_cal = render(calib_prompts_l, None)

    b_last = collect_pooled_activations(
        model, tok, benign_fit, steer_layers, position="last",
        batch_size=batch_size, prefix_cache=teacher_prefix_cache).acts
    m_last = collect_pooled_activations(
        model, tok, malicious_fit, steer_layers, position="last",
        batch_size=batch_size, prefix_cache=teacher_prefix_cache).acts

    rs, refusal_norms = {}, {}
    for l in steer_layers:
        raw = b_last[l].mean(0) - m_last[l].mean(0)        # +r = "add the SI"
        refusal_norms[l] = float(raw.norm())
        rs[l] = raw / raw.norm().clamp(min=1e-12)          # unit

    pad_id = tok.pad_token_id
    lam_budget = float(lam) if lam is not None else 1.0
    beta_used: float | None = beta
    beta_derived = False
    fit_acts = s_fit = cal_rows = cal_acts = s_cal = cal_idx = None
    position_subsample: dict | None = None

    if stage == 1:
        rule = ({"nullspace_ratio": nullspace_ratio}
                if nullspace_ratio is not None else {"tau": tau})
        ws = {l: fit_stage1(m_last[l], b_last[l], alpha=alpha, **rule)
              for l in steer_layers}
    elif stage == 2 and beta is not None:
        # The pinned-beta leg, bit-identical to the pre-derivation stage-2
        # path — for deliberately reproducing a recorded step size. The
        # DEFAULT (beta None) derives beta in the branch below.
        ws = {l: fit_stage2(m_last[l], b_last[l], beta=beta, alpha=alpha,
                            center_gram=center_gram,
                            mean_shift_weight=mean_shift_weight)
              for l in steer_layers}
    elif stage == 2:
        # Stage 2, beta None -> DERIVE beta against the eps budget, exactly
        # the way stage 3 does (2026-08-24; module docstring carries the
        # full why + the measured monotonicity caveats). The FIT penalty is
        # untouched — the last-token unit-weight Gram, fit_stage2's exact
        # arithmetic, built once because it is beta-independent — while the
        # CERTIFICATE the bisection prices against is stage 3's: predicted
        # benign KL on a held-out Fisher-weighted calibration harvest. The
        # certificate is a property of the resulting object (ws + deployed
        # gate + lam) and the model, not of which penalty produced the ws,
        # so the stage-2 fit is substitutable under the same kl_fn — and
        # the Fisher weights are what price it in nats/position, making
        # eps mean the same thing on every rung of the ladder.
        Mb_fit = {l: benign_penalty(b_last[l],
                                    torch.ones(b_last[l].shape[0]),
                                    center_gram=center_gram,
                                    mean_shift_weight=mean_shift_weight)
                  for l in steer_layers}
        cal_rows = rollout_ids(model, tok, benign_cal,
                               max_new_tokens=rollout_tokens,
                               batch_size=batch_size)
        cal_acts, s_cal, cal_idx = harvest_positions(
            model, cal_rows, steer_layers, pad_id=pad_id, writes=rs,
            batch_size=batch_size, fisher_mode=fisher_mode,
            mc_samples=mc_samples, seed=seed + 1,
            max_positions=resolved_cap)
        position_subsample = {
            "max_capture_positions": resolved_cap,
            "cap_source": cap_source,
            "capture_budget_bytes": (DEFAULT_CAPTURE_BUDGET_BYTES
                                     if cap_source == "auto" else None),
            "calib": _subsample_record(cal_rows, cal_idx, seed + 1),
        }
        if position_subsample["calib"]["capped"]:
            r = position_subsample["calib"]
            print(f"[alphakl] stage 2: calib capture CAPPED at "
                  f"{resolved_cap} — kept "
                  f"{r['kept_positions']}/{r['total_positions']} "
                  f"positions across {r['pairs_contributing']}/"
                  f"{r['pairs_total']} rows (seed {r['seed']})")

        def fit_at(b):
            return {l: ridge(m_last[l], Mb_fit[l], b, alpha)
                    for l in steer_layers}

        def kl_at(ws_, lam_):
            return predicted_benign_kl(ws_, lam_, cal_acts, s_cal, cal_idx,
                                       clamp=clamp_gate)["mean"]

        beta_used, ws = calibrate_beta(
            fit_at, lambda w_: kl_at(w_, lam_budget), eps,
            iters=beta_iters)
        beta_derived = True
        print(f"[alphakl] stage 2: beta={beta_used:.4g} derived from "
              f"eps={eps:g} nats/position at lam={lam_budget:g}")
    else:
        fit_rows = rollout_ids(model, tok, benign_fit,
                               max_new_tokens=rollout_tokens,
                               batch_size=batch_size)
        if unprompted_texts and include_unprompted_in_penalty:
            fit_rows = fit_rows + text_ids(tok, unprompted_texts)
        fit_acts, s_fit, fit_idx = harvest_positions(
            model, fit_rows, steer_layers, pad_id=pad_id, writes=rs,
            batch_size=batch_size, fisher_mode=fisher_mode,
            mc_samples=mc_samples, seed=seed,
            max_positions=resolved_cap)
        cal_rows = rollout_ids(model, tok, benign_cal,
                               max_new_tokens=rollout_tokens,
                               batch_size=batch_size)
        cal_acts, s_cal, cal_idx = harvest_positions(
            model, cal_rows, steer_layers, pad_id=pad_id, writes=rs,
            batch_size=batch_size, fisher_mode=fisher_mode,
            mc_samples=mc_samples, seed=seed + 1,
            max_positions=resolved_cap)

        # The capture cap's provenance — see _subsample_record for the
        # field semantics; both legs recorded, a capped fit must never
        # look like an uncapped one.
        position_subsample = {
            "max_capture_positions": resolved_cap,
            "cap_source": cap_source,
            "capture_budget_bytes": (DEFAULT_CAPTURE_BUDGET_BYTES
                                     if cap_source == "auto" else None),
            "fit": _subsample_record(fit_rows, fit_idx, seed),
            "calib": _subsample_record(cal_rows, cal_idx, seed + 1),
        }
        for leg in ("fit", "calib"):
            r = position_subsample[leg]
            if r["capped"]:
                print(f"[alphakl] stage 3: {leg} capture CAPPED at "
                      f"{position_subsample['max_capture_positions']} — kept "
                      f"{r['kept_positions']}/{r['total_positions']} "
                      f"positions across {r['pairs_contributing']}/"
                      f"{r['pairs_total']} rows (seed {r['seed']})")

        # The penalty Gram is beta-INDEPENDENT: build it once per layer and
        # bisect over ridge() directly. Recomputing it inside fit_stage3 on
        # every bisection iteration is 40x a [T_total, d] x [d, d] matmul —
        # invisible at a short SI, hours of CPU at a several-thousand-token
        # system prompt.
        Mb_fit = {l: benign_penalty(fit_acts[l], s_fit,
                                    center_gram=center_gram,
                                    mean_shift_weight=mean_shift_weight)
                  for l in steer_layers}

        def fit_at(b):
            return {l: ridge(m_last[l], Mb_fit[l], b, alpha)
                    for l in steer_layers}

        def kl_at(ws_, lam_):
            return predicted_benign_kl(ws_, lam_, cal_acts, s_cal, cal_idx,
                                       clamp=clamp_gate)["mean"]

        if beta is None:
            beta_used, ws = calibrate_beta(
                fit_at, lambda w_: kl_at(w_, lam_budget), eps,
                iters=beta_iters)
            beta_derived = True
            print(f"[alphakl] stage 3: beta={beta_used:.4g} derived from "
                  f"eps={eps:g} nats/position at lam={lam_budget:g}")
        else:
            ws = fit_at(beta)

    obj = AlphaKLObject(model, steer_layers, ws, rs,
                        base_scale=lam_budget, clamp_gate=clamp_gate,
                        stage=stage)
    obj.to_(next(model.parameters()).device)

    calib_info: dict = {}
    if lam is None:
        def _steered_generate(scale: float) -> list[str]:
            return generate_with_object(
                model, tok, malicious_cal, obj=obj, scale=scale,
                batch_size=batch_size, max_new_tokens=calib_max_new_tokens)

        calib_info = calibrate_base_scale(
            model, tok, malicious_cal, _steered_generate,
            scale_ladder=scale_ladder, rep_threshold=rep_threshold,
            ppl_guard_factor=ppl_guard_factor,
            max_new_tokens=calib_max_new_tokens, batch_size=batch_size,
            log_fn=log_fn, arm=f"alphakl{stage}",
        )
        obj.base_scale = calib_info["base_scale"]
        if beta_derived:   # stages 2-3: certify at the scale that deploys
            beta_used, ws = calibrate_beta(
                fit_at, lambda w_: kl_at(w_, obj.base_scale), eps,
                iters=beta_iters)
            obj.ws = {int(l): w.detach().float() for l, w in ws.items()}
            obj.to_(next(model.parameters()).device)
            print(f"[alphakl] stage {stage}: beta re-bisected at the "
                  f"calibrated scale {obj.base_scale:g} -> "
                  f"beta={beta_used:.4g}")

    # ---- diagnostics: probe quality + the mandatory KL verification ----
    b_cal_last = collect_pooled_activations(
        model, tok, benign_cal, steer_layers, position="last",
        batch_size=batch_size, prefix_cache=teacher_prefix_cache).acts
    m_cal_last = collect_pooled_activations(
        model, tok, malicious_cal, steer_layers, position="last",
        batch_size=batch_size, prefix_cache=teacher_prefix_cache).acts
    gate_b = _gates(obj.ws, b_cal_last, clamp_gate)
    gate_m = _gates(obj.ws, m_cal_last, clamp_gate)
    probe = {
        "probe_auc": _rank_auc(gate_m, gate_b),
        "gate_malicious_mean": float(gate_m.mean()),
        "gate_benign_mean": float(gate_b.mean()),
    }

    if cal_rows is None:  # stage 3 and derived-beta stage 2 already rolled
        cal_rows = rollout_ids(model, tok, benign_cal,
                               max_new_tokens=rollout_tokens,
                               batch_size=batch_size)
    warnings = list(calib_info.get("warnings", []))
    benign_gate: dict | None = None
    if lam is None:
        # The benign-KL operating-point gate (the 2026-08-22 wave fix; full
        # rationale on cap_scale_to_benign_budget): the canary picked the
        # scale, so the harness — not the caller — is accountable for the
        # operating point actually honoring the ladder's benign contract.
        budget = (BENIGN_KL_BUDGET_FACTOR * eps if benign_kl_budget is None
                  else float(benign_kl_budget))
        canary_scale = obj.base_scale
        results: dict[float, dict] = {}

        def _measure(cand: float) -> float:
            got = measured_benign_kl(
                model, obj, cal_rows, pad_id=pad_id,
                scale=cand / canary_scale, batch_size=batch_size)
            results[cand] = got
            return got["mean"]

        final_scale, gate_trail, gate_capped = cap_scale_to_benign_budget(
            _measure, base_scale=canary_scale, scale_ladder=scale_ladder,
            budget=budget, arm=f"alphakl{stage}")
        measured = results[final_scale]
        obj.base_scale = final_scale
        benign_gate = {
            "budget": budget,
            "budget_source": ("default_2x_eps" if benign_kl_budget is None
                              else "configured"),
            "canary_scale": canary_scale,
            "trail": gate_trail,
            "capped": gate_capped,
        }
        if gate_capped:
            warnings.append(
                f"benign-KL gate CAPPED the operating scale: canary picked "
                f"{canary_scale:g} but its measured benign KL "
                f"{gate_trail[0]['kl_measured_prompted']:.4g} nats/position "
                f"exceeds the budget {budget:g}; shipping {final_scale:g} "
                f"({gate_trail[-1]['kl_measured_prompted']:.4g} "
                "nats/position)")
            print(f"[alphakl] WARNING: {warnings[-1]}")
        if log_fn is not None:
            for i, t in enumerate(gate_trail):
                log_fn({f"alphakl{stage}_klgate/scale": t["scale"],
                        f"alphakl{stage}_klgate/kl_measured_prompted":
                            t["kl_measured_prompted"]}, step=i)
    else:
        measured = measured_benign_kl(model, obj, cal_rows, pad_id=pad_id,
                                      scale=1.0, batch_size=batch_size)
    kl_info = {"kl_measured_prompted": measured["mean"],
               "kl_lam_operating": obj.base_scale}
    if cal_acts is not None:   # a certificate basis exists: stage 3 always,
        predicted = predicted_benign_kl(   # stage 2 when beta was derived
            obj.ws, obj.base_scale, cal_acts, s_cal, cal_idx,
            clamp=clamp_gate)
        kl_info["kl_predicted_prompted"] = predicted["mean"]
        ratio = measured["mean"] / max(predicted["mean"], 1e-12)
        kl_info["kl_measured_over_predicted"] = ratio
        if not 0.5 <= ratio <= 2.0:
            warnings.append(
                f"measured/predicted benign KL = {ratio:.2f} (outside the "
                "spec's 2x band) — the second-order budget is off at this "
                "operating point; consider bisecting beta on MEASURED KL")
            print(f"[alphakl] WARNING: {warnings[-1]}")
    if unprompted_texts:
        un_rows = text_ids(tok, unprompted_texts)
        kl_info["kl_measured_unprompted"] = measured_benign_kl(
            model, obj, un_rows, pad_id=pad_id, scale=1.0,
            batch_size=batch_size)["mean"]
        kl_info["unprompted_in_penalty"] = bool(include_unprompted_in_penalty)

    obj.metadata = {
        "spec": SPEC,
        "stage": stage,
        "layers": steer_layers,
        "n_pairs": len(prompts),
        "n_fit": len(fit_prompts),
        "n_calib": n_calib,
        "alpha": alpha,
        "tau": (tau if stage == 1 and nullspace_ratio is None else None),
        "nullspace_ratio": (nullspace_ratio if stage == 1 else None),
        "beta": beta_used,
        "beta_derived_from_eps": beta_derived,
        # eps binds wherever a certificate was issued: stage 3 always,
        # stage 2 when beta was derived against it. fisher_mode priced the
        # certificate whenever a Fisher-weighted harvest ran (cal_acts).
        "eps": (eps if (stage == 3 or beta_derived) else None),
        "lam_given": lam,
        "benign_kl_gate": benign_gate,
        "clamp_gate": clamp_gate,
        "center_gram": center_gram,
        "fisher_mode": (fisher_mode if cal_acts is not None else None),
        "rollout_tokens": (rollout_tokens
                           if (stage == 3 or beta_derived) else None),
        "position_subsample": position_subsample,
        "refusal_norms": refusal_norms,
        **probe,
        **kl_info,
        **{k: v for k, v in calib_info.items() if k != "warnings"},
        "warnings": warnings,
    }
    if log_fn is not None:
        log_fn({f"alphakl{stage}/{k}": v for k, v in kl_info.items()
                if isinstance(v, (int, float))}, step=0)
    return obj
