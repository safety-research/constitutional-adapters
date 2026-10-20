"""AlphaSteer (arXiv 2506.07022, ICLR 2026) as an SI steering arm.

FIDELITY SOURCE: the authors' code, github.com/AlphaLab-USTC/AlphaSteer
@ 3cd0d2e (verified 2026-08-17). AlphaSteer learns, in CLOSED FORM (no
gradient descent), a per-layer d×d matrix M = P · Δ̃ and applies an
INPUT-DEPENDENT edit h += strength · (h_last · M):

- P = null-space projector over BENIGN activations. SVD of the benign Gram
  Aᵀ·A; Q = the bottom ``nullspace_ratio·d`` right singular vectors (the
  directions of LEAST benign variance); P = Q·Qᵀ. So P·h ≈ 0 for a benign
  activation → the learned edit vanishes on benign inputs (verified in code:
  steering_utils.cal_null_space_project, rank rule = a FRACTION of the
  hidden dim, not an energy tolerance).
- Δ̃ = ridge least-squares mapping the PROJECTED malicious activations onto
  the refusal direction: Δ̃ = pinv(Xᵀ·X + λ·P) · (Xᵀ·R), X = H_mal·P,
  R = the (per-layer) refusal vector stacked over the malicious rows,
  λ = 10 (steering_utils.cal_tilde_delta_with_regularization_l,
  calc_steering_matrix.py lambda_reg=10.0).
- refusal direction = a difference-in-means (paper Eq. 2, Arditi-style;
  the authors ship a precomputed pickle, README credits AdaSteer's
  extraction).
- inference: PREFILL ONLY (gated on seq_len > 1), edit computed from the
  last valid PROMPT token and added to every prompt position; decode steps
  are influenced only through the steered prompt KV-cache.

DOCUMENTED DIVERGENCES (paper/code vs our calls):
1. LAYERS. The paper steers a 10-layer subset ([8,9,10,11,12,13,14,16,18,19]
   for Llama-3.1-8B). We pin the SINGLE campaign layer 15 (block-15 output,
   the harness convention) for apples-to-apples comparability with every
   other arm — a 4096×4096 matrix, ~67 MB fp32, vs the paper's ~10×.
   (The multi-layer, paper-faithful form lives beside it below — see
   MULTI-LAYER PORT; the single-layer pin stays the default for every arm
   that does not opt in through its own config key.)
2. STRENGTH. The paper sweeps a small NEGATIVE grid tied to its shipped
   pickle's sign. Our refusal direction is recomputed with an explicit sign
   (below) so POSITIVE strength steers toward SI-compliance; the absolute
   magnitude has no paper rule, so a base scale is calibrated on the canary
   (distinct-3 + 3x-gen-ppl) and the eval factor grid multiplies it.
3. SITE. The paper edits resid_pre of each steered layer; we capture AND
   edit at resid_post of block 15 (the harness's "layer 15"), capture site
   and apply site matched.

GENERALIZATION: extraction/alphakl.py carries this method's rank-1
factorization onward — the target matrix R stacks copies of r, so the d x d
solution factors as Delta = u r' exactly, and the hard null-space constraint
becomes stage 1 of a three-stage ladder (soft covariance penalty, then
Fisher/KL weighting under a benign-KL budget). This file stays the
paper-faithful arm; parity between the two stage-1 forms is tested in
tests/test_alphakl.py.

SI STEERING ADAPTATION (the benign/malicious mapping):
- "benign" (steering must VANISH — the null-space set) → inputs where the SI
  is already satisfied: the SI-PROMPTED (system-present) renderings. The
  null-space constraint then makes the compiled edit ≈ 0 on prompted inputs,
  so the `both` (SI + vector) arm composes cleanly — directly addressing the
  Stage-1 finding that additive vectors DEGRADE a working system prompt.
- "malicious" (steering must FIRE) → inputs where compliance requires
  steering: the UNPROMPTED (system-absent) renderings (the eval
  distribution).
- refusal direction R → the SI direction mu(SI-present) − mu(SI-absent):
  the projected unprompted activation is mapped toward "as if the SI were
  present". Whether the edit fires strongly is EMPIRICAL — if SI-present and
  SI-absent differ mainly along benign-high-variance directions, P will damp
  the edit; the null-space rank/energy diagnostics record this so a weak
  result is legible, not silent.

MULTI-LAYER PORT (2026-09-08; ported from an earlier development
branch @ 6a1ba1df). The paper's deployment —
several steered layers, per-layer null-space ratios, resid_pre site,
prefill-only — is ``train_alphasteer_content_multilayer`` +
``AlphaSteerObject``'s keyword-only ``layers=``/``Ms=`` form. The port is
CONDITIONED ON CONSTRUCTION so the landed single-layer path is unchanged
byte for byte (spec: "existing Llama Class B behaviour is byte-identical"):

- ``ridge_delta`` keeps ``torch.linalg.pinv`` (SVD) as its DEFAULT solver.
  Only the multi-layer trainer selects the eigh-based ``psd_pinv``, which
  it needs: ``A = XᵀX + λP`` has rank exactly ``rank(P)``, so at d = 4096
  the ``d − k`` exactly-repeated zero eigenvalues stall SVD's bidiagonal
  iteration ("too many repeated singular values" — the real 10-layer Llama
  fit crashed there). The single-layer fits never produce that spectrum at
  the sizes they run at, and keeping their solver keeps their numbers.
- An object built through the legacy positional signature
  (``layer=``, ``M=``) records that lineage and ``save()`` writes MAIN's
  five-key blob in main's order. An object built with ``layers=``/``Ms=``
  writes the plural blob (``layers``/``Ms``/``refusals``/``site``/
  ``steer_positions``) with the legacy singular keys present and ``None`` —
  the form ``capeval.serving.lens_adapters._adapt_alphasteer`` reads.
- ``_collect_last_acts`` tokenizes the ALREADY chat-rendered inputs with
  ``add_special_tokens=False``: the template has
  emitted the BOS already, and vLLM's chat path tokenizes the render the
  same way, so the default ``True`` would fit on a DOUBLE-BOS prompt the
  server never produces (Llama-3.1: ``[128000, 128000, 128006, …]`` vs the
  served ``[128000, 128006, …]``; measured on all five campaign tokenizers,
  Llama is the one affected). A ``CaptureIdVerifier`` asserts the forwarded
  ids equal the serving render's ids — on IDS, since a template-text hash
  cannot see this. The legacy single-layer path captures through
  ``capture.collect_pooled_activations`` and keeps its behaviour (fixing
  the defect there is a separate, deliberate change).
- The forward under capture asks for ONE logit position
  (``capture._last_logit_kwargs``): value-identical for hooked hidden states
  and ~2 GB/batch cheaper at a 262k vocab.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ..models import build_chat_prompt
from .calibrate import DEFAULT_SCALE_LADDER, calibrate_base_scale
from .capture import (
    _last_logit_kwargs,
    capture_layer_outputs,
    collect_pooled_activations,
)
from .fit_checks import check_sample_floor

PAPER = "arXiv:2506.07022 (AlphaSteer, ICLR 2026)"
DEFAULT_NULLSPACE_RATIO = 0.5   # paper Llama-3.1 uses 0.4–0.6 per layer
DEFAULT_LAMBDA_REG = 10.0       # calc_steering_matrix.py lambda_reg

STEER_POSITIONS = ("prefill", "all", "prompt_hold")
SITES = ("post", "pre")
#: ridge_delta's pseudo-inverse solvers. "svd" is torch.linalg.pinv — the
#: landed single-layer behaviour and the DEFAULT; "psd_pinv" is the eigh
#: inverse the multi-layer trainer needs (see the module docstring).
RIDGE_SOLVERS = ("svd", "psd_pinv")
#: prompts whose ids the capture asserts equal to the serving render's
#: (CaptureIdVerifier); the one default the trainers, the screen and the
#: fit driver's rule block all read.
DEFAULT_VERIFY_SAMPLE = 32

# The paper's Llama-3.1-8B-Instruct deployment (src/utils/const.py:34-44
# @ 3cd0d2e, verified 2026-08-18): 10 steered layers — 15 and 17 are
# DELIBERATELY skipped — with a PER-LAYER null-space ratio. Exported for
# orchestrators registering the paper-faithful multi-layer arm, and for the
# Llama layer-rule recovery gate (plan Phase D), which compares against them.
PAPER_LLAMA31_LAYERS = [8, 9, 10, 11, 12, 13, 14, 16, 18, 19]
PAPER_LLAMA31_NULLSPACE_RATIOS = {
    8: 0.6, 9: 0.6, 10: 0.6, 11: 0.6, 12: 0.4,
    13: 0.5, 14: 0.6, 16: 0.6, 18: 0.6, 19: 0.6,
}


def last_valid_token_index(
    attn_mask: torch.Tensor | None, batch: int, seq_len: int, device
) -> torch.Tensor:
    """Index of the LAST attended (mask == 1) position per row — robust to
    left- AND right-padded batches, matching the reference repo's
    mask_utils.get_last_valid_token_index. With no usable 2-D mask (None,
    or a shape that does not cover this forward — e.g. a KV-cache decode
    step whose mask spans past+current) every row falls back to
    ``seq_len - 1``, which is exactly the legacy ``hidden[:, -1, :]``
    behavior (and IS the last real token under left padding)."""
    if (attn_mask is None or attn_mask.dim() != 2
            or attn_mask.shape[0] != batch or attn_mask.shape[1] != seq_len):
        return torch.full((batch,), seq_len - 1, dtype=torch.long,
                          device=device)
    m = attn_mask.to(device=device).long()
    return (seq_len - 1) - m.flip(-1).argmax(dim=-1)


def _eigh_with_jitter(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.linalg.eigh(A)`` (ascending) with the last-resort diagonal
    jitter for a pathologically conditioned symmetric input — a tiny shift
    cannot move the null space but rescues convergence. The ONE copy:
    ``null_space_projector``, ``psd_pinv`` and the screen's shared-eigh loop
    all call it, so "a prefix slice equals its projector bit for bit" holds
    by construction rather than by transcription."""
    try:
        return torch.linalg.eigh(A)
    except (torch.linalg.LinAlgError, RuntimeError):
        d = A.shape[0]
        jitter = 1e-6 * float(A.diagonal().mean().clamp(min=1e-8))
        return torch.linalg.eigh(
            A + jitter * torch.eye(d, dtype=A.dtype, device=A.device))


def projector_from_prefix(evecs: torch.Tensor, ratio: float) -> tuple[torch.Tensor, int]:
    """``P = Q Qᵀ`` from the first ``max(1, int(ratio·d))`` columns of an
    ASCENDING eigenvector matrix (the least-variance directions), plus the
    rank ``k``. The one rank rule: ``null_space_projector`` applies it to its
    own eigh, the screen applies it to one shared eigh per layer for every
    p in the grid — the same function, so the two cannot drift."""
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"nullspace_ratio must be in (0, 1), got {ratio}")
    d = evecs.shape[0]
    k = max(1, int(ratio * d))
    Q = evecs[:, :k]
    return Q @ Q.t(), k


def null_space_projector(
    benign: torch.Tensor, ratio: float
) -> tuple[torch.Tensor, dict]:
    """P over the benign activation matrix ``benign`` [N, d]: the bottom
    ``ratio·d`` least-variance directions of the benign Gram (Aᵀ·A), P = Q·Qᵀ.
    Faithful to cal_null_space_project (eigenvectors of Aᵀ·A, fraction-of-d
    rank rule). Returns (P [d, d], diagnostics).

    ROBUSTNESS FIX (jobs 141778_{11,23,26} — number_words/number_bullets/title
    × alphasteer FAILED here): the benign Gram is LOW-RANK — n_prompts (≈200)
    ≪ d (4096), so it has ~d−n REPEATED (≈0) eigenvalues, and
    torch.linalg.svd's iterative bidiagonalization does not converge on a
    spectrum that degenerate (LinAlgError "too many repeated singular values";
    it happened to converge for json_format in the canary, but the whole arm
    was one bad draw from crashing). torch.linalg.eigh is the correct solver
    for a SYMMETRIC PSD matrix and is robust to a degenerate spectrum; its
    eigenvectors ARE the right singular vectors of A, so the null-space P is
    identical wherever svd DID converge (verified: the diffmean-style
    diagnostics are unchanged on the passing cells). Ascending eigenvalues ⇒
    the FIRST k eigenvectors are the least-benign-variance (null-space)
    directions."""
    A = benign.float()
    d = A.shape[1]
    gram = A.t() @ A
    evals, evecs = _eigh_with_jitter(gram)   # ascending; evecs[:, i] ↔ evals[i]
    P, k = projector_from_prefix(evecs, ratio)
    benign_energy = float((A @ P).pow(2).sum() / (A.pow(2).sum() + 1e-8))
    info = {
        "nullspace_ratio": ratio,
        "nullspace_rank": int(k),
        "hidden_dim": int(d),
        "benign_energy_kept": benign_energy,  # ~0 ⇒ benign well nulled
        "gram_top_eigenvalue": float(evals[-1]),
        "gram_bottom_eigenvalue": float(evals[0]),
    }
    return P, info


def psd_pinv(
    A: torch.Tensor, rcond: float | None = None
) -> tuple[torch.Tensor, dict]:
    """Moore–Penrose pseudo-inverse of a SYMMETRIC PSD matrix via eigh —
    never SVD.

    ROBUSTNESS FIX (the second call site of the f22998f failure class —
    the REAL 10-layer content fit on Llama-3.1-8B crashed in ridge_delta
    with torch._C._LinAlgError "linalg.svd: ... too many repeated singular
    values"): ridge_delta's A = XᵀX + λ·P has rank exactly k = rank(P)
    (X = H·P lives entirely inside range(P)), so the orthogonal complement
    carries d−k EXACTLY repeated zero eigenvalues — the same degenerate
    spectrum torch.linalg.svd's iterative bidiagonalization fails to
    converge on at d=4096. torch.linalg.pinv is SVD-based, so it inherits
    the crash; torch.linalg.eigh is the correct solver for a symmetric
    matrix and is robust to a degenerate spectrum. For a symmetric PSD
    matrix the SVD and the eigendecomposition COINCIDE (singular values =
    eigenvalues, singular vectors = eigenvectors), so the truncated-eigh
    inverse is the SAME operator pinv computes wherever pinv DID converge
    (asserted to tight tolerance in the tests).

    PORT DECISION — the truncation rule mirrors torch.linalg.pinv's own
    default rtol: rcond = d · eps(compute dtype) (fp32 here ⇒ ~4.9e-4 at
    d=4096), and eigenvalues ≤ rcond · λ_max are inverted to 0. Negative
    eigenvalues (fp noise on a PSD input) are clamped to 0 first, so they
    always truncate. An all-zero matrix pseudo-inverts to zero. Should
    eigh itself ever fail, the same last-resort diagonal jitter as
    null_space_projector rescues convergence.

    Reached ONLY from the multi-layer trainer (``ridge_delta(...,
    solver="psd_pinv")``); the legacy single-layer fits keep
    torch.linalg.pinv so their numbers do not move.

    Returns (pinv(A) [d, d], diagnostics)."""
    A = A.float()
    d = A.shape[0]
    if rcond is None:
        rcond = d * torch.finfo(A.dtype).eps
    evals, evecs = _eigh_with_jitter(A)         # ascending
    evals = evals.clamp(min=0.0)
    keep = evals > rcond * float(evals[-1])
    inv = torch.zeros_like(evals)
    inv[keep] = evals[keep].reciprocal()
    pinv = (evecs * inv.unsqueeze(0)) @ evecs.t()
    info = {
        "pinv_rcond": float(rcond),
        "pinv_rank": int(keep.sum()),
        "pinv_top_eigenvalue": float(evals[-1]),
    }
    return pinv, info


def _ridge_system(H: torch.Tensor, P: torch.Tensor, r: torch.Tensor,
                  lambda_reg: float):
    """``X = H P``, ``R`` = ``r`` stacked, ``A = XᵀX + λP``, ``b = XᵀR`` —
    the ridge's normal equations (cal_tilde_delta_with_regularization_l).
    Shared by :func:`ridge_delta` and the screen loop, which also needs
    ``pinv(A)`` for the rank-1 factor and so cannot call ``ridge_delta``."""
    X = H @ P                                   # [N, d]
    R = r.unsqueeze(0).expand(X.shape[0], -1)   # [N, d]
    A_mat = X.t() @ X + lambda_reg * P          # P symmetric idempotent ⇒ PᵀP=P
    b_mat = X.t() @ R                           # [d, d]
    return X, R, A_mat, b_mat


def _ridge_info(H: torch.Tensor, X: torch.Tensor, R: torch.Tensor,
                tilde_delta: torch.Tensor, lambda_reg: float) -> dict:
    """The ridge's diagnostics (main's exact key set and order)."""
    resid = float((X @ tilde_delta - R).norm() / (R.norm() + 1e-8))
    mal_energy = float(X.pow(2).sum() / (H.pow(2).sum() + 1e-8))   # X = H P
    return {
        "lambda_reg": lambda_reg,
        "regression_residual": resid,     # ‖XΔ̃ − R‖ / ‖R‖
        "malicious_energy_kept": mal_energy,  # > benign_energy_kept ⇒ fires
    }


def ridge_delta(
    malicious: torch.Tensor,
    P: torch.Tensor,
    refusal: torch.Tensor,
    lambda_reg: float,
    rcond: float | None = None,
    *,
    solver: str = "svd",
) -> tuple[torch.Tensor, dict]:
    """Δ̃ = pinv(Xᵀ·X + λ·P)·(Xᵀ·R), X = H_mal·P, R = refusal stacked.
    Faithful to cal_tilde_delta_with_regularization_l. Returns
    (tilde_delta [d, d], diagnostics).

    ``solver="svd"`` (DEFAULT) is ``torch.linalg.pinv(A) @ b`` — the landed
    single-layer expression, unchanged, with main's exact diagnostics key
    set. ``solver="psd_pinv"`` pseudo-inverts through :func:`psd_pinv`
    (eigh, torch.linalg.pinv's default truncation rule) and adds its
    ``pinv_*`` diagnostics — A_mat is symmetric PSD with ~d−rank(P) exactly
    repeated zero eigenvalues, the spectrum SVD-based pinv fails to converge
    on at scale; the multi-layer trainer selects it. ``rcond`` is a
    psd_pinv knob and refuses under the svd solver rather than being
    silently ignored."""
    if solver not in RIDGE_SOLVERS:
        raise ValueError(f"solver must be one of {RIDGE_SOLVERS}, got {solver!r}")
    if solver == "svd" and rcond is not None:
        raise ValueError("rcond applies to solver='psd_pinv' only; "
                         "torch.linalg.pinv keeps its own default rtol")
    H = malicious.float()
    r = refusal.float()
    X, R, A_mat, b_mat = _ridge_system(H, P, r, lambda_reg)
    if solver == "svd":
        tilde_delta = torch.linalg.pinv(A_mat) @ b_mat
        pinv_info: dict = {}
    else:
        A_pinv, pinv_info = psd_pinv(A_mat, rcond=rcond)
        tilde_delta = A_pinv @ b_mat
    info = {**_ridge_info(H, X, R, tilde_delta, lambda_reg), **pinv_info}
    return tilde_delta, info


class AlphaSteerObject:
    """Trained AlphaSteer artifact behind the stage-2 object protocol —
    SINGLE-LAYER (the landed campaign shape) or MULTI-LAYER (the
    paper's deployment). Construct with either the original positional
    signature::

        AlphaSteerObject(model, layer, M, base_scale, refusal)   # unchanged

    or the keyword-only multi-layer form::

        AlphaSteerObject(model, base_scale=1.0,
                         layers=[8, 9, ...], Ms={8: M8, 9: M9, ...},
                         refusals={8: r8, ...},          # optional
                         steer_positions="prefill", site="post")

    Exactly one of (``layer``+``M``) or (``layers``+``Ms``) must be given.
    ``self.layer`` / ``self.M`` / ``self.refusal`` stay readable on a
    single-layer object (None on multi-layer); the normalized state is
    always ``self.layers`` / ``self.Ms`` / ``self.refusals``.

    CONSTRUCTION LINEAGE decides the blob ``save()`` writes: the
    legacy signature at the legacy deployment (``site="post"``,
    ``steer_positions="prefill"``) writes MAIN's five-key blob byte for
    byte; everything else writes the plural form. Two objects with the same
    matrices therefore serialise differently depending on how they were
    built — deliberately, because a pre-port single-layer fit and a
    post-port one must produce the SAME FILE.

    steer_positions — WHEN the edit applies (default "prefill" = the
    landed behavior, bit-identical):

    - "prefill": only forwards with seq_len > 1 are edited; the vector
      base_scale*scale*(h_last @ M_l) comes from the LAST VALID prompt
      token and is BROADCAST to every prompt position; decode steps
      (seq_len == 1) untouched (the paper's deployment).
    - "all": the PRINCIPLED PER-POSITION generalization — at EVERY
      position, prompt and decode alike (including seq_len == 1 KV-cache
      steps), add base_scale*scale*(h_t @ M_l) computed from THAT
      position's own hidden state. Deployment-matched for comparison
      against per-position gated steering arms.
    - "prompt_hold": prefill behaves exactly like "prefill", and the SAME
      fixed prompt-derived vector is then re-added at every decode
      position (the alternative reading of "extend AlphaSteer to the
      response": hold the prompt edit rather than recompute it). The held
      vector refreshes on each new prefill inside one apply(); a decode
      step before any prefill is untouched.

    site — WHERE the edit applies (default "post" = the landed harness
    convention): "post" edits the residual stream LEAVING block l (forward
    hook); "pre" edits the stream ENTERING block l, before its
    input_layernorm (forward-pre hook) — the paper's resid_pre site.
    resid_pre of block l == resid_post of block l-1.

    Last-token robustness: the "last valid token" index is derived from
    the 2-D attention mask captured off the top-level model call (see
    ``last_valid_token_index``), correct for BOTH left- and right-padded
    batches. Without a mask, or unpadded, it falls back to seq_len-1 —
    exactly the legacy ``hidden[:, -1, :]``, so unpadded and left-padded
    calls (the generation path) are bit-identical to the landed
    implementation; right-padded batches are a correctness FIX (the old
    code read a pad position's hidden state there).

    Memory: each M is d x d. Llama-3.1-8B (d=4096) = 64 MiB fp32 per
    layer; the paper's 10-layer deployment ~640 MiB (~671 MB) fp32 —
    ``save()`` always stores fp32 on CPU, so that is the artifact size.
    At apply time each M is lazily cast to the model's device/dtype and
    the cast CACHED (cleared by ``to_``), so a bf16 model steady-states at
    ~320 MiB of bf16 copies for 10 layers; if GPU memory is tight, keep
    the fp32 masters on CPU (skip ``to_(cuda)``) and only the cached bf16
    casts live on the GPU.
    """

    kind = "alphasteer"

    def __init__(self, model, layer: int | None = None,
                 M: torch.Tensor | None = None,
                 base_scale: float = 1.0,
                 refusal: torch.Tensor | None = None,
                 metadata: dict | None = None, *,
                 layers: list[int] | None = None,
                 Ms: dict[int, torch.Tensor] | None = None,
                 refusals: dict[int, torch.Tensor] | None = None,
                 steer_positions: str = "prefill",
                 site: str = "post"):
        from ..models import get_decoder_layers

        if steer_positions not in STEER_POSITIONS:
            raise ValueError(f"steer_positions must be one of "
                             f"{STEER_POSITIONS}, got {steer_positions!r}")
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}, got {site!r}")
        multi = layers is not None or Ms is not None
        single = layer is not None or M is not None
        if multi == single:
            raise ValueError(
                "give EITHER the single-layer (layer, M) pair OR the "
                "multi-layer (layers, Ms) pair — not both, not neither")
        if multi:
            if layers is None or Ms is None:
                raise ValueError(
                    "multi-layer form needs BOTH layers and Ms")
            if refusal is not None:
                raise ValueError("multi-layer form takes per-layer "
                                 "`refusals`, not a single `refusal`")
            self.layers = [int(l) for l in layers]
            if not self.layers:
                raise ValueError("layers must be non-empty")
            if len(set(self.layers)) != len(self.layers):
                raise ValueError(f"duplicate layers in {self.layers}")
            Ms = {int(l): m for l, m in Ms.items()}
            missing = [l for l in self.layers if l not in Ms]
            if missing:
                raise ValueError(f"Ms missing matrices for layers {missing}")
            self.Ms = {l: Ms[l].detach().float() for l in self.layers}
            self.refusals = {int(l): r.detach().float()
                             for l, r in (refusals or {}).items()}
        else:
            if layer is None or M is None:
                raise ValueError("single-layer form needs BOTH layer and M")
            if refusals is not None:
                raise ValueError("single-layer form takes `refusal`, "
                                 "not per-layer `refusals`")
            self.layers = [int(layer)]
            self.Ms = {int(layer): M.detach().float()}
            self.refusals = ({} if refusal is None
                             else {int(layer): refusal.detach().float()})
        #: construction lineage — True iff built through the legacy
        #: positional (layer, M) signature. Read by save() (see the class
        #: docstring); never changes after construction.
        self._legacy_form = not multi
        self.base_scale = float(base_scale)
        self.steer_positions = steer_positions
        self.site = site
        self.metadata = metadata or {}
        self._decoder = get_decoder_layers(model)
        self._Mcast: dict[tuple, torch.Tensor] = {}

    # -- single-layer views (the legacy attribute surface; None on multi) --

    @property
    def layer(self) -> int | None:
        return self.layers[0] if len(self.layers) == 1 else None

    @property
    def M(self) -> torch.Tensor | None:
        return self.Ms[self.layers[0]] if len(self.layers) == 1 else None

    @property
    def refusal(self) -> torch.Tensor | None:
        return (self.refusals.get(self.layers[0])
                if len(self.layers) == 1 else None)

    @property
    def legacy_form(self) -> bool:
        """True iff constructed through the legacy (layer, M) signature —
        the lineage that makes ``save()`` write main's five-key blob."""
        return self._legacy_form

    @property
    def trainable(self) -> list:
        return []  # closed-form; nothing trains by gradient

    def n_params(self) -> int:
        return int(sum(m.numel() for m in self.Ms.values()))

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    # -- the edit ----------------------------------------------------------

    def _cast_M(self, layer: int, device, dtype) -> torch.Tensor:
        """M_l on (device, dtype), cached so "all"-mode decode loops don't
        re-cast 64 MiB per token per layer. Cleared by ``to_``."""
        key = (layer, device, dtype)
        got = self._Mcast.get(key)
        if got is None:
            got = self.Ms[layer].to(device=device, dtype=dtype)
            self._Mcast[key] = got
        return got

    def _edit(self, layer: int, hidden: torch.Tensor, mag: float,
              state: dict) -> torch.Tensor:
        """The layer's edit on ``hidden`` [B, S, d]; returns ``hidden``
        ITSELF (same object) when this forward must not be steered."""
        B, S, _ = hidden.shape
        mode = self.steer_positions
        if mode == "all":
            M = self._cast_M(layer, hidden.device, hidden.dtype)
            return hidden + (mag * (hidden @ M)).to(hidden.dtype)
        if S <= 1:                              # a decode step
            if mode == "prompt_hold":
                add = state["hold"].get(layer)
                if add is None:
                    return hidden               # no prefill seen yet
                if add.shape[0] != B:
                    raise RuntimeError(
                        f"prompt_hold: decode batch ({B}) != prompt batch "
                        f"({add.shape[0]}) — the held prompt vector cannot "
                        "be matched to rows (beam expansion unsupported)")
                return hidden + add.to(
                    device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
            return hidden                       # "prefill": decode untouched
        # a prompt (seq_len > 1) forward
        M = self._cast_M(layer, hidden.device, hidden.dtype)
        idx = last_valid_token_index(state.get("mask"), B, S, hidden.device)
        h_last = hidden[torch.arange(B, device=hidden.device), idx]  # [B, d]
        add = mag * (h_last @ M)                                     # [B, d]
        if mode == "prompt_hold":
            state["hold"][layer] = add.detach()
        return hidden + add.to(hidden.dtype).unsqueeze(1)   # every prompt pos

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mag = self.base_scale * scale
        if mag == 0.0:
            yield        # exact no-op: no hooks registered at all
            return
        state: dict = {"mask": None, "hold": {}}
        handles = []
        if self.steer_positions != "all":
            # h_last needs the 2-D attention mask, which layer hooks never
            # see — capture it off the top-level model call ("all" edits
            # per position and never indexes a last token).
            def mask_hook(_mod, args, kwargs):
                m = kwargs.get("attention_mask")
                if m is None and len(args) > 1 and torch.is_tensor(args[1]):
                    m = args[1]     # forward(input_ids, attention_mask, ...)
                state["mask"] = (m if m is not None and m.dim() == 2
                                 else None)
                return None

            handles.append(model.register_forward_pre_hook(
                mask_hook, with_kwargs=True))

        def make_post(layer):
            def hook(_mod, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                steered = self._edit(layer, hidden, mag, state)
                if steered is hidden:
                    return output
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered
            return hook

        def make_pre(layer):
            def hook(_mod, args, kwargs):
                if "hidden_states" in kwargs:
                    hidden = kwargs["hidden_states"]
                elif args and torch.is_tensor(args[0]):
                    hidden = args[0]
                else:
                    return None
                steered = self._edit(layer, hidden, mag, state)
                if steered is hidden:
                    return None
                if "hidden_states" in kwargs:
                    return (args, {**kwargs, "hidden_states": steered})
                return ((steered,) + tuple(args[1:]), kwargs)
            return hook

        for l in self.layers:
            if self.site == "post":
                handles.append(
                    self._decoder[l].register_forward_hook(make_post(l)))
            else:
                handles.append(self._decoder[l].register_forward_pre_hook(
                    make_pre(l), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def to_(self, device) -> None:
        self.Ms = {l: m.to(device) for l, m in self.Ms.items()}
        self.refusals = {l: r.to(device) for l, r in self.refusals.items()}
        self._Mcast.clear()

    def _writes_legacy_blob(self) -> bool:
        """The legacy five-key blob is written iff the object was built
        through the legacy signature AND sits at the legacy deployment
        (post / prefill) — the only configuration main could express, so
        the only one whose file must match main's byte for byte. A legacy-
        built object moved to ``site="pre"`` or another steer_positions has
        no legacy representation and round-trips through the plural form
        instead of silently dropping those fields."""
        return (self._legacy_form and self.site == "post"
                and self.steer_positions == "prefill")

    def save(self, path: Path) -> None:
        """fp32-on-CPU blob. Legacy lineage (see ``_writes_legacy_blob``)
        writes MAIN's exact key set in main's order — {kind, layer, M,
        base_scale, refusal, metadata} — so a pre-port and a post-port
        single-layer fit produce sha256-identical files. Every other object
        writes the plural form: layers/Ms/refusals/steer_positions/site with
        the legacy singular keys present and None (what
        lens_adapters._adapt_alphasteer reads). 10-layer Llama-8B: ~640 MiB
        on disk."""
        if self._writes_legacy_blob():
            l0 = self.layers[0]
            refusal = self.refusals.get(l0)
            torch.save(
                {"kind": self.kind, "layer": l0,
                 "M": self.Ms[l0].detach().float().cpu(),
                 "base_scale": self.base_scale,
                 "refusal": (None if refusal is None
                             else refusal.detach().float().cpu()),
                 "metadata": self.metadata},
                path,
            )
            return
        torch.save(
            {"kind": self.kind,
             "layer": None,
             "M": None,
             "refusal": None,
             "layers": list(self.layers),
             "Ms": {l: m.detach().float().cpu()
                    for l, m in self.Ms.items()},
             "refusals": ({l: r.detach().float().cpu()
                           for l, r in self.refusals.items()} or None),
             "steer_positions": self.steer_positions,
             "site": self.site,
             "base_scale": self.base_scale,
             "metadata": self.metadata},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "AlphaSteerObject":
        # Missing fields default to the LEGACY values, so a blob saved
        # before multi-layer existed (single layer/M, no steer_positions/
        # site keys) loads and behaves exactly as it always did — and,
        # having been rebuilt through the legacy signature, re-saves as
        # main's five-key blob.
        steer_positions = blob.get("steer_positions", "prefill")
        site = blob.get("site", "post")
        if blob.get("Ms"):
            refusals = blob.get("refusals") or None
            return cls(
                model, base_scale=float(blob["base_scale"]),
                metadata=blob.get("metadata", {}),
                layers=[int(l) for l in blob["layers"]],
                Ms={int(l): m for l, m in blob["Ms"].items()},
                refusals=(None if refusals is None else
                          {int(l): r for l, r in refusals.items()}),
                steer_positions=steer_positions, site=site)
        return cls(
            model, layer=int(blob["layer"]), M=blob["M"],
            base_scale=float(blob["base_scale"]),
            refusal=blob.get("refusal"),
            metadata=blob.get("metadata", {}),
            steer_positions=steer_positions, site=site)


def train_alphasteer(
    model,
    tok,
    instruction,
    pairs,
    *,
    layer: int,
    log_fn=None,
    nullspace_ratio: float = DEFAULT_NULLSPACE_RATIO,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    batch_size: int = 8,
    calib_prompts: int = 16,
    calib_max_new_tokens: int = 128,
    scale_ladder=DEFAULT_SCALE_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    epochs: int | None = None,  # train_kwargs.default compat (documented no-op)
    max_len: int | None = None,  # train_kwargs.default compat (documented no-op)
) -> AlphaSteerObject:
    """Registry trainer: capture benign (SI-present) + malicious (SI-absent)
    last-token activations, build the null-space projector + ridge matrix in
    closed form, then calibrate the base scale on the canary procedure.
    epochs/max_len are documented no-ops (closed-form extraction)."""
    del epochs, max_len
    prompts = [p.prompt for p in pairs]
    if not prompts:
        raise ValueError("AlphaSteer extraction got an empty pairs list")

    benign_texts = [build_chat_prompt(tok, p, system=instruction.text)
                    for p in prompts]           # SI present ⇒ null-space set
    malicious_texts = [build_chat_prompt(tok, p, system=None)
                       for p in prompts]        # SI absent ⇒ steering fires
    benign = collect_pooled_activations(
        model, tok, benign_texts, [layer], position="last",
        batch_size=batch_size).acts[layer]
    malicious = collect_pooled_activations(
        model, tok, malicious_texts, [layer], position="last",
        batch_size=batch_size).acts[layer]

    refusal = benign.mean(dim=0) - malicious.mean(dim=0)  # +R = "add the SI"
    # n-vs-d floor (2026-08-20): both closed-form pieces are d-dimensional
    # per row — the null space needs >= d benign rows for a full-rank Gram,
    # the ridge maps n_malicious d-vectors onto r. n < d fits are legal
    # (the ridge regularizes) but must be LABELLED, never silent.
    d_fit = int(benign.shape[1])
    floor_info = {
        "sample_floor": [
            check_sample_floor(int(benign.shape[0]), d_fit,
                               what="null-space rows (benign)",
                               tag="alphasteer", log_fn=log_fn),
            check_sample_floor(int(malicious.shape[0]), d_fit,
                               what="ridge rows (malicious)",
                               tag="alphasteer", log_fn=log_fn),
        ]
    }
    P, ns_info = null_space_projector(benign, nullspace_ratio)
    tilde_delta, reg_info = ridge_delta(malicious, P, refusal, lambda_reg)
    M = P @ tilde_delta                                    # [d, d]

    obj = AlphaSteerObject(model, layer, M, base_scale=1.0, refusal=refusal)
    obj.to_(next(model.parameters()).device)

    calib_rendered = malicious_texts[:calib_prompts]

    def _steered_generate(scale: float) -> list[str]:
        from ..generate import generate_with_object

        return generate_with_object(
            model, tok, calib_rendered, obj=obj, scale=scale,
            batch_size=batch_size, max_new_tokens=calib_max_new_tokens,
        )

    calib = calibrate_base_scale(
        model, tok, calib_rendered, _steered_generate,
        scale_ladder=scale_ladder, rep_threshold=rep_threshold,
        ppl_guard_factor=ppl_guard_factor,
        max_new_tokens=calib_max_new_tokens, batch_size=batch_size,
        log_fn=log_fn, arm="alphasteer",
    )
    obj.base_scale = calib["base_scale"]
    obj.metadata = {
        "paper": PAPER,
        "layer": layer,
        "n_pairs": len(prompts),
        "refusal_norm": float(refusal.norm()),
        "matrix_frobenius": float(M.norm()),
        **floor_info,
        **ns_info,
        **reg_info,
        **{k: v for k, v in calib.items() if k != "warnings"},
        "warnings": calib["warnings"],
    }
    return obj


# ---------------------------------------------------------------------------
# SI-AGNOSTIC fit-once mode (content-based; the paper-faithful setting)
# ---------------------------------------------------------------------------
# train_alphasteer (above) is the SI steering ADAPTATION: benign = SI-present
# renderings, malicious = SI-absent, r = the "add-the-SI" direction — one
# object per SI. The paper fits ONCE on harmful-vs-benign CONTENT with a
# precomputed refusal direction, SI-agnostically. train_alphasteer_content
# is that path (sets from data/content_sets.py).


def refusal_direction(
    harmful: torch.Tensor, benign: torch.Tensor
) -> torch.Tensor:
    """The base "refusals-vs-compliances" direction: diff-in-means of
    last-token activations, harmful − benign (Arditi-style; what AdaSteer's
    extraction and AlphaSteer's shipped per-layer pickle encode). Harmful
    prompts elicit refusal, benign elicit compliance, so +r = "add refusal".
    Not unit-normalized — the ridge maps the projected malicious activations
    onto it, exactly as with the SI-mode ``refusal`` vector."""
    return harmful.float().mean(dim=0) - benign.float().mean(dim=0)


def train_alphasteer_content(
    model,
    tok,
    benign_prompts,
    harmful_prompts,
    *,
    layer: int,
    refusal: torch.Tensor | None = None,
    log_fn=None,
    nullspace_ratio: float = DEFAULT_NULLSPACE_RATIO,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    batch_size: int = 8,
    calib_prompts: int = 16,
    calib_max_new_tokens: int = 128,
    scale_ladder=DEFAULT_SCALE_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
) -> AlphaSteerObject:
    """Fit AlphaSteer ONCE on labeled harmful/benign CONTENT, SI-agnostically
    (the paper's setting). ``benign_prompts`` build the null space (P);
    ``harmful_prompts`` are the ridge regression source (X = H_harmful·P
    mapped onto the refusal direction r). Both are rendered SI-FREE
    (system=None), so the object carries no SI and applies to any SI
    comparison.

    r = ``refusal`` if given (a precomputed per-model refusal vector), else
    the base refusals-vs-compliances diff-in-means over these same sets.

    Calibrated on the harmful (fire) set — the SI-mode's malicious analogue —
    since the edit vanishes on benign by construction and a benign-only
    calibration never brackets incoherence."""
    if not benign_prompts or not harmful_prompts:
        raise ValueError("alphasteer content fit needs non-empty benign and "
                         "harmful prompt lists")
    benign_texts = [build_chat_prompt(tok, p, system=None)
                    for p in benign_prompts]
    harmful_texts = [build_chat_prompt(tok, p, system=None)
                     for p in harmful_prompts]
    benign = collect_pooled_activations(
        model, tok, benign_texts, [layer], position="last",
        batch_size=batch_size).acts[layer]
    harmful = collect_pooled_activations(
        model, tok, harmful_texts, [layer], position="last",
        batch_size=batch_size).acts[layer]

    r = refusal_direction(harmful, benign) if refusal is None else refusal.float()
    # n-vs-d floor (2026-08-20): the 41-harmful HarmBench-val default once
    # fit this d x d ridge at d = 4096 with no complaint. The authors' own
    # composition (2,000 harmful) is still n < d — legal under the ridge,
    # but it must be labelled, with the realised n and d named.
    d_fit = int(benign.shape[1])
    floor_info = {
        "sample_floor": [
            check_sample_floor(int(benign.shape[0]), d_fit,
                               what="null-space rows (benign)",
                               tag="alphasteer_content", log_fn=log_fn),
            check_sample_floor(int(harmful.shape[0]), d_fit,
                               what="ridge rows (harmful)",
                               tag="alphasteer_content", log_fn=log_fn),
        ]
    }
    P, ns_info = null_space_projector(benign, nullspace_ratio)
    tilde_delta, reg_info = ridge_delta(harmful, P, r, lambda_reg)
    M = P @ tilde_delta

    obj = AlphaSteerObject(model, layer, M, base_scale=1.0, refusal=r)
    obj.to_(next(model.parameters()).device)

    calib_rendered = harmful_texts[:calib_prompts]

    def _steered_generate(scale: float) -> list[str]:
        from ..generate import generate_with_object

        return generate_with_object(
            model, tok, calib_rendered, obj=obj, scale=scale,
            batch_size=batch_size, max_new_tokens=calib_max_new_tokens,
        )

    calib = calibrate_base_scale(
        model, tok, calib_rendered, _steered_generate,
        scale_ladder=scale_ladder, rep_threshold=rep_threshold,
        ppl_guard_factor=ppl_guard_factor,
        max_new_tokens=calib_max_new_tokens, batch_size=batch_size,
        log_fn=log_fn, arm="alphasteer_content",
    )
    obj.base_scale = calib["base_scale"]
    obj.metadata = {
        "paper": PAPER,
        "fit_mode": "content_si_agnostic",
        "layer": layer,
        "n_benign": len(benign_prompts),
        "n_harmful": len(harmful_prompts),
        "refusal_given": refusal is not None,
        "refusal_norm": float(r.norm()),
        "matrix_frobenius": float(M.norm()),
        **floor_info,
        **ns_info,
        **reg_info,
        **{k: v for k, v in calib.items() if k != "warnings"},
        "warnings": calib["warnings"],
    }
    return obj


# ---------------------------------------------------------------------------
# MULTI-LAYER SI-agnostic fit (the paper's full deployment)
# ---------------------------------------------------------------------------


def serving_render_ids(tok, rendered: str) -> list[int]:
    """The token ids the SERVER produces for an already chat-rendered
    string: the render tokenized with ``add_special_tokens=False`` (vLLM's
    ``ChatCompletionRequest.add_special_tokens`` default — "the chat
    template takes care of adding the special tokens"). The reference every
    capture forward is checked against, on IDS."""
    return list(tok(rendered, add_special_tokens=False)["input_ids"])


class CaptureIdMismatch(ValueError):
    """The ids a capture forward was about to consume differ from the
    serving render's ids — the double-BOS class of defect. Raised BEFORE
    the forward, so no activation is ever captured on a representation the
    server does not produce."""


class CaptureIdVerifier:
    """Asserts, for the first ``n_sample`` rendered prompts a capture pass
    encodes, that the ids about to be forwarded (batch-encoded, padded, then
    un-padded through the attention mask) equal :func:`serving_render_ids`
    of the same prompt — an INDEPENDENT single-text call, so a batch-path
    post-processor that prepends a spurious special token (the tokenizer's
    default ``add_special_tokens=True`` on a template that already emitted
    BOS) is caught, not recorded. A template-text hash cannot see this; the
    check is on ids by design.

    ``summary()`` is what lands in ``fit_meta.json["capture"]``: the count
    verified and a sha256 over the sampled reference ids."""

    def __init__(self, tok, n_sample: int = DEFAULT_VERIFY_SAMPLE):
        if n_sample < 1:
            raise ValueError("n_sample must be >= 1")
        self.tok = tok
        self.n_sample = int(n_sample)
        self.n_checked = 0
        self._hasher = hashlib.sha256()

    def check(self, texts: list[str], enc) -> None:
        if self.n_checked >= self.n_sample:
            return
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        for j, text in enumerate(texts):
            if self.n_checked >= self.n_sample:
                break
            forwarded = ids[j][mask[j].bool()].tolist()
            reference = serving_render_ids(self.tok, text)
            if forwarded != reference:
                k = next((i for i, (a, b) in enumerate(zip(forwarded, reference))
                          if a != b), min(len(forwarded), len(reference)))
                raise CaptureIdMismatch(
                    "capture ids differ from the serving render's ids "
                    f"(prompt #{self.n_checked}, first difference at position "
                    f"{k}): about to forward {len(forwarded)} ids with head "
                    f"{forwarded[:4]}, the server tokenizes this render to "
                    f"{len(reference)} ids with head {reference[:4]}. A "
                    "leading duplicate is the double-BOS defect (plan §A "
                    "deviation 2): the chat template already emitted the "
                    "special tokens, so the capture must tokenize with "
                    "add_special_tokens=False. Refusing to capture on a "
                    "representation the server never produces.")
            self._hasher.update(json.dumps(reference).encode("utf-8"))
            self._hasher.update(b"\n")
            self.n_checked += 1

    def summary(self) -> dict:
        return {
            "add_special_tokens": False,
            "ids_match_serving_render": True,   # a mismatch RAISES; it is never recorded
            "n_verified": int(self.n_checked),
            "sampled_render_ids_sha256": self._hasher.hexdigest(),
            "note": ("chat-rendered inputs tokenized with add_special_tokens="
                     "False, matching vLLM's ChatCompletionRequest default; "
                     "ids asserted equal to tok(render, add_special_tokens="
                     "False) on the sampled prompts — on Llama-3.1 the "
                     "tokenizer default would prepend a SECOND BOS"),
        }


#: progress cadence of _collect_last_acts: one log line per this many
#: batches (at the campaign's width 8 that is every 400 rows -- ~35 lines for
#: a 14,000-row side). Logging only; the captured bytes do not depend on it.
CAPTURE_LOG_EVERY_BATCHES = 50


def _collect_last_acts(
    model, tok, texts: list[str], layers: list[int], site: str,
    batch_size: int, *, add_special_tokens: bool = False,
    verifier: CaptureIdVerifier | None = None,
    log_fn=None,
) -> dict[int, torch.Tensor]:
    """Last-VALID-token activations {layer: [N, d]} at ``site``, captured
    for ALL layers in ONE forward pass per batch (never re-forwarding per
    layer). Right-pads locally and indexes by the attention mask, so the
    pooled row is the true last prompt token. The local sibling of
    capture.collect_pooled_activations(position="last"), which is pinned
    to site="post"; capture site here matches the deploy site (the same
    capture/apply-site matching the single-layer arm documents).

    ``texts`` are ALREADY chat-rendered, so the default
    ``add_special_tokens=False`` is the serving tokenization (see the module
    docstring) — ``True`` is kept only so the branch-parity test can pin the
    port at the branch's tokenization. ``verifier`` (a
    :class:`CaptureIdVerifier`) checks each batch's ids against the serving
    render BEFORE it is forwarded. The forward keeps one logit position
    (``_last_logit_kwargs``): value-identical for the hooked hidden states,
    and the difference between a 64-row batch fitting and OOMing at a 262k
    vocab.

    ``log_fn`` (optional) receives ONE progress line every
    :data:`CAPTURE_LOG_EVERY_BATCHES` batches -- rows done, elapsed, rate --
    so a healthy slow capture and a hung one are distinguishable from the
    job log (job 470950 sat silent for 21 minutes between "split written"
    and a cancel). It is handed counts and timings only, never a tensor, and
    the forward, the hooks and the pooled rows are identical with or without
    it (test-pinned)."""
    device = next(model.parameters()).device
    out: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    prev_side = tok.padding_side
    tok.padding_side = "right"
    t0 = time.time()
    n_batches = 0
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          add_special_tokens=add_special_tokens)
                if verifier is not None:
                    verifier.check(chunk, enc)
                enc = enc.to(device)
                mask = enc["attention_mask"]
                rows = torch.arange(mask.shape[0], device=mask.device)
                last = mask.sum(dim=1) - 1          # right-padded here
                with capture_layer_outputs(model, list(layers),
                                           site=site) as cap:
                    model(**enc, **_last_logit_kwargs(model))
                for l in layers:
                    out[l].append(cap[l][rows, last].float().cpu())
                n_batches += 1
                done = i + len(chunk)
                if log_fn is not None and n_batches % CAPTURE_LOG_EVERY_BATCHES == 0 \
                        and done < len(texts):
                    el = time.time() - t0
                    log_fn(f"capture {done}/{len(texts)} rows ({n_batches} batches) "
                           f"in {el:.0f}s -- {done / max(el, 1e-9):.1f} rows/s")
    finally:
        tok.padding_side = prev_side
    return {l: torch.cat(v, dim=0) for l, v in out.items()}


def train_alphasteer_content_multilayer(
    model,
    tok,
    benign_prompts,
    harmful_prompts,
    *,
    layers: list[int],
    nullspace_ratios: dict[int, float] | float = DEFAULT_NULLSPACE_RATIO,
    refusals: dict[int, torch.Tensor] | None = None,
    steer_positions: str = "prefill",
    site: str = "post",
    log_fn=None,
    lambda_reg: float = DEFAULT_LAMBDA_REG,
    batch_size: int = 8,
    calib_prompts: int = 16,
    calib_max_new_tokens: int = 128,
    scale_ladder=DEFAULT_SCALE_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    add_special_tokens: bool = False,
    verify_capture_ids: bool | None = None,
    verify_sample: int = DEFAULT_VERIFY_SAMPLE,
    calibrate: bool = True,
    precaptured: tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]] | None = None,
) -> AlphaSteerObject:
    """train_alphasteer_content generalized to the paper's MULTI-LAYER
    deployment: one closed-form (P_l, tilde_delta_l) fit per steered layer
    with PER-LAYER null-space ratios, activations for ALL steered layers
    captured in ONE forward pass per batch, at the SAME site the object
    will deploy at (capture/apply-site matched).

    ``nullspace_ratios``: one float shared by every layer, or a per-layer
    dict that must cover every entry of ``layers`` — the paper's Llama-3.1
    values are PAPER_LLAMA31_NULLSPACE_RATIOS (they vary by layer: 0.4 at
    12, 0.5 at 13, 0.6 elsewhere). ``refusals``: optional per-layer
    precomputed refusal vectors (the paper ships a pickle); any missing
    layer falls back to that layer's harmful-vs-benign diff-in-means over
    these same sets. ``lambda_reg`` defaults to the paper's 10.0.

    The ridge pseudo-inverse is :func:`psd_pinv` here (``ridge_delta(...,
    solver="psd_pinv")``): a 10-layer d=4096 fit produces exactly the
    repeated-zero spectrum that stalls SVD (module docstring).

    Capture tokenizes the chat-rendered prompts with
    ``add_special_tokens=False`` (the serving tokenization) and, unless
    ``verify_capture_ids`` is False (default: verify whenever
    ``add_special_tokens`` is False), asserts the first ``verify_sample``
    prompts' ids equal the serving render's ids before forwarding; the
    check's summary lands in ``metadata["capture"]``. ``precaptured =
    (benign_acts, harmful_acts)`` — ``{layer: [N, d]}`` dicts already
    captured at ``site`` for these exact prompts (the F.9 screen's acts
    dir) — skips the capture pass; every steered layer must be present.

    ONE base_scale is calibrated for the whole stack and applied
    identically at every steered layer, matching the paper (a single
    strength across its 10 layers; 0 elsewhere). Calibrated on the harmful
    (fire) set, like the single-layer content fit. ``calibrate=False``
    leaves ``base_scale = 1.0`` (the unscaled object) for a caller whose
    strength is set elsewhere — the served lambda ladder (plan Phase C) —
    so no unrecorded factor is baked twice.

    Artifact size: d x d fp32 per layer — ~640 MiB (~671 MB) for the
    paper's 10-layer Llama-8B deployment (see AlphaSteerObject.save)."""
    if not benign_prompts or not harmful_prompts:
        raise ValueError("alphasteer content fit needs non-empty benign and "
                         "harmful prompt lists")
    steer_layers = [int(l) for l in layers]
    if not steer_layers:
        raise ValueError("layers must be a non-empty list")
    if len(set(steer_layers)) != len(steer_layers):
        raise ValueError(f"duplicate layers in {steer_layers}")
    if isinstance(nullspace_ratios, dict):
        ratios = {int(l): float(r) for l, r in nullspace_ratios.items()}
        missing = [l for l in steer_layers if l not in ratios]
        if missing:
            raise ValueError(
                f"nullspace_ratios is missing layers {missing} — give a "
                "ratio for every steered layer, or one float for all")
    else:
        ratios = {l: float(nullspace_ratios) for l in steer_layers}
    if verify_capture_ids is None:
        verify_capture_ids = not add_special_tokens

    benign_texts = [build_chat_prompt(tok, p, system=None)
                    for p in benign_prompts]
    harmful_texts = [build_chat_prompt(tok, p, system=None)
                     for p in harmful_prompts]
    if precaptured is not None:
        benign, harmful = precaptured
        for name, acts, n in (("benign", benign, len(benign_prompts)),
                              ("harmful", harmful, len(harmful_prompts))):
            missing = [l for l in steer_layers if l not in acts]
            if missing:
                raise ValueError(
                    f"precaptured {name} activations lack layers {missing}")
            bad = [l for l in steer_layers if int(acts[l].shape[0]) != n]
            if bad:
                raise ValueError(
                    f"precaptured {name} activations have "
                    f"{[int(acts[l].shape[0]) for l in bad]} rows at layers "
                    f"{bad}, but {n} {name} prompts were given — the acts "
                    "must be captured on exactly these prompts, in order")
        benign = {l: benign[l].float() for l in steer_layers}
        harmful = {l: harmful[l].float() for l in steer_layers}
        capture_info = {"source": "precaptured", "add_special_tokens": None,
                        "ids_match_serving_render": None,
                        "note": "activations supplied by the caller (the F.9 "
                                "screen's acts dir); its own capture record "
                                "carries the ids check"}
    else:
        verifier = (CaptureIdVerifier(tok, n_sample=verify_sample)
                    if verify_capture_ids else None)
        benign = _collect_last_acts(model, tok, benign_texts, steer_layers,
                                    site, batch_size,
                                    add_special_tokens=add_special_tokens,
                                    verifier=verifier)
        harmful = _collect_last_acts(model, tok, harmful_texts, steer_layers,
                                     site, batch_size,
                                     add_special_tokens=add_special_tokens,
                                     verifier=verifier)
        if verifier is not None:
            capture_info = {"source": "captured", **verifier.summary()}
        else:
            capture_info = {"source": "captured",
                            "add_special_tokens": bool(add_special_tokens),
                            "ids_match_serving_render": None,
                            "note": "ids check disabled by the caller"}

    # n-vs-d floor (2026-08-20), kept from the single-layer trainers: d is
    # shared by every layer, so one pair of records labels the whole fit.
    d_fit = int(benign[steer_layers[0]].shape[1])
    floor_info = {
        "sample_floor": [
            check_sample_floor(int(benign[steer_layers[0]].shape[0]), d_fit,
                               what="null-space rows (benign)",
                               tag="alphasteer_content_ml", log_fn=log_fn),
            check_sample_floor(int(harmful[steer_layers[0]].shape[0]), d_fit,
                               what="ridge rows (harmful)",
                               tag="alphasteer_content_ml", log_fn=log_fn),
        ]
    }

    Ms: dict[int, torch.Tensor] = {}
    rs: dict[int, torch.Tensor] = {}
    per_layer: dict[int, dict] = {}
    for l in steer_layers:
        r = (refusals[l].float() if refusals is not None and l in refusals
             else refusal_direction(harmful[l], benign[l]))
        P, ns_info = null_space_projector(benign[l], ratios[l])
        tilde_delta, reg_info = ridge_delta(harmful[l], P, r, lambda_reg,
                                            solver="psd_pinv")
        Ms[l] = P @ tilde_delta
        rs[l] = r
        per_layer[l] = {
            "refusal_norm": float(r.norm()),
            "matrix_frobenius": float(Ms[l].norm()),
            **ns_info,
            **reg_info,
        }

    obj = AlphaSteerObject(model, base_scale=1.0, layers=steer_layers,
                           Ms=Ms, refusals=rs,
                           steer_positions=steer_positions, site=site)
    obj.to_(next(model.parameters()).device)

    if calibrate:
        calib_rendered = harmful_texts[:calib_prompts]

        def _steered_generate(scale: float) -> list[str]:
            from ..generate import generate_with_object

            return generate_with_object(
                model, tok, calib_rendered, obj=obj, scale=scale,
                batch_size=batch_size, max_new_tokens=calib_max_new_tokens,
            )

        calib = calibrate_base_scale(
            model, tok, calib_rendered, _steered_generate,
            scale_ladder=scale_ladder, rep_threshold=rep_threshold,
            ppl_guard_factor=ppl_guard_factor,
            max_new_tokens=calib_max_new_tokens, batch_size=batch_size,
            log_fn=log_fn, arm="alphasteer_content_ml",
        )
        obj.base_scale = calib["base_scale"]
        calib_meta = {**{k: v for k, v in calib.items() if k != "warnings"},
                      "warnings": calib["warnings"],
                      "base_scale_source": "calibrate_base_scale"}
    else:
        calib_meta = {"base_scale": 1.0, "warnings": [],
                      "base_scale_source": "uncalibrated_fit (strength set "
                                           "by the caller, e.g. the served "
                                           "lambda ladder)"}
    obj.metadata = {
        "paper": PAPER,
        "fit_mode": "content_si_agnostic_multilayer",
        "layers": steer_layers,
        "steer_positions": steer_positions,
        "site": site,
        "lambda_reg": lambda_reg,
        "ridge_solver": "psd_pinv",
        "nullspace_ratios": ratios,
        "n_benign": len(benign_prompts),
        "n_harmful": len(harmful_prompts),
        "refusals_given": sorted(refusals) if refusals else [],
        "per_layer": per_layer,
        "capture": capture_info,
        **floor_info,
        **calib_meta,
    }
    return obj
