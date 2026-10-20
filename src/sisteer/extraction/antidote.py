"""Jailbreak Antidote (arXiv 2410.02298, Shen et al., ICLR 2025) as an
SI steering arm: per-layer PCA safety-direction extraction + sparse
last-token residual adjustment h' = h + alpha * (d ⊙ m).

FIDELITY SOURCES (verified 2026-08-17 against the arXiv v2 HTML and the
authors' own code — Beijing-AISI/panda-guard, whose README checkmarks
"JailbreakAntidote" and whose RepeDefender is the implementation vehicle:
rep_token=-1, PCA direction, `calc_topk` |.|-max binary mask WITHOUT
renormalization, injected vector = ctrl_factor * direction * sign):

Paper equations implemented here (task ruling: paper over code where the
two differ):
- direction: PCA over last-token hidden states of the POOLED contrast set,
  C^l = (1/|H|) sum (h - h_bar)(h - h_bar)^T, d^l = first principal
  component (unit norm by construction).
- sparsify: binary mask m keeping the top k% of d's dimensions by
  |component| (paper default k = 5%); the masked direction is NOT
  renormalized (the paper has none).
- update: h' = h + alpha (d ⊙ m) at the LAST TOKEN position only —
  hooks.py positions="last": the final real prompt token during prefill,
  every incremental forward's single position during KV-cached decode.

DOCUMENTED DIVERGENCES between the paper's equations and the authors'
panda-guard code, and our calls:
1. panda-guard reaches PCA through RepE's pipeline with n_difference=1
   (PCA over pairwise DIFFERENCES of hidden states); the paper's written
   covariance pools raw states without differencing. We implement the
   paper's equation.
2. The paper states no sign convention for the PC. panda-guard signs it
   via RepE's label-projection agreement. We use the equivalent mean-based
   rule: flip the PC iff its projection onto (mean_target - mean_other)
   is negative, so +alpha always moves toward the TARGET cluster.
3. The paper applies the shift at a layer subset L (panda-guard: top 37.5%
   of layers by rep-reading significance). We pin the SINGLE layer 15 of
   Llama-3.1-8B — a comparability pin, matching the `vector` arm.

SI STEERING ADAPTATIONS (deliberate, mirroring every other arm):
- contrast: SI-prompted vs UNPROMPTED renderings of the arm's standard A1
  pair prompts (the paper contrasts benign vs harmful prompts); hidden
  states at the last PROMPT token, pre-generation (Stolfo instr_diff's
  capture, PCA instead of mean-diff).
- alpha: the PC is unit-norm and alpha is ABSOLUTE — no norm-relative
  machinery (a documented difference from the `vector` arm's
  norm_relative scaling). The paper picks per-model alpha ranges by
  manual coherence inspection; we automate it: a geometric ladder swept
  under the harness degeneration guard (distinct-3 < rep_threshold, or
  steered-GENERATION perplexity under the unsteered model > 3x the
  unsteered generations' — the standard 3x-ppl guard scored on
  generations because a positions="last" full forward's NLL is blind to
  the intervention: only the final position is steered and its logits
  drop out of the shifted NLL). base_alpha = the midpoint of the coherent
  range [0, alpha_hi]; the eval factor grid multiplies base_alpha.
"""

from __future__ import annotations

import torch

from ..hooks import steering
from ..vectors import SteeringVector
from .capture import collect_pooled_activations
from .fit_checks import check_sample_floor

PAPER = "arXiv:2410.02298v2 (Jailbreak Antidote, Shen et al., ICLR 2025)"

# Geometric default ladder: the masked unit-PC has norm <= 1 while
# Llama-8B mid-layer residual norms are O(10) — the paper's 7-8B ranges
# top out single-digit (Qwen2-7B [-7, 5]), so 0.5..64 brackets the
# incoherence bound with room on both sides.
DEFAULT_ALPHA_LADDER = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def antidote_direction(
    target: torch.Tensor, other: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """First principal component of the POOLED (target ∪ other) states,
    sign-aligned so +alpha moves toward the TARGET cluster.

    target/other: [N, d] pooled last-token activations (float32 CPU).
    Returns (pc1 [d], info) with explained-variance ratio and sign
    evidence; a near-zero sign projection or a weak PC1 is WARNED loudly
    and recorded (metadata is the audit trail).
    """
    X = torch.cat([target, other], dim=0).float()
    if X.shape[0] < 2:
        raise ValueError(
            f"antidote PCA needs >= 2 pooled examples, got {X.shape[0]}"
        )
    Xc = X - X.mean(dim=0)
    # C = (1/|H|) Xc^T Xc; PC1 = first right singular vector of Xc.
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    pc1 = Vh[0]
    total = float((S**2).sum())
    evr = float(S[0] ** 2) / total if total > 0 else 0.0

    delta = target.float().mean(dim=0) - other.float().mean(dim=0)
    proj = float(delta @ pc1)
    # cosine between the mean-difference and the PC: the sign EVIDENCE.
    sign_cos = proj / (float(delta.norm()) + 1e-8)
    flipped = proj < 0
    if flipped:
        pc1 = -pc1
    warnings = []
    if evr < 0.05:
        warnings.append(
            f"PC1 explains only {evr:.4f} of pooled variance — the "
            "contrast may not be the dominant direction"
        )
    if abs(sign_cos) < 0.05:
        warnings.append(
            f"sign projection is near zero (cos {sign_cos:.4f}) — the "
            "PC barely separates the clusters; the sign choice is fragile"
        )
    for w in warnings:
        print(f"[antidote] WARNING: {w}")
    info = {
        "explained_variance_ratio_pc1": evr,
        "sign_projection": proj,
        "sign_cos": sign_cos,
        "sign_flipped": bool(flipped),
        "warnings": warnings,
    }
    return pc1, info


def sparsify(direction: torch.Tensor, k_frac: float) -> tuple[torch.Tensor, dict]:
    """Binary top-k% |component| mask (paper Eq.: m_i = 1 iff |d_i| >= tau).

    NOT renormalized after masking — the paper has no normalization, and
    panda-guard applies ctrl_factor to the masked vector directly.
    k = int(k_frac * d) mirrors panda-guard's calc_topk floor.
    """
    if not 0.0 < k_frac <= 1.0:
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
    d = direction.shape[-1]
    k = max(1, int(k_frac * d))
    _, idx = torch.topk(direction.abs(), k)
    mask = torch.zeros_like(direction, dtype=torch.bool)
    mask[idx] = True
    masked = direction * mask
    info = {
        "k_frac": k_frac,
        "k_dims": int(k),
        "masked_norm": float(masked.norm()),
    }
    return masked, info


def calibrate_base_alpha(
    model,
    tok,
    rendered_prompts: list[str],
    direction: torch.Tensor,
    layer: int,
    *,
    alpha_ladder=DEFAULT_ALPHA_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    max_new_tokens: int = 128,
    batch_size: int = 8,
    log_fn=None,
) -> dict:
    """Paper-style alpha range finding, automated: walk the ladder upward,
    guard each rung with distinct-3 + the 3x-ppl guard scored on the
    steered GENERATIONS under the unsteered model (see module docstring
    for why a positions="last" forward-NLL guard would be blind), and
    take base_alpha = midpoint of the coherent range [0, alpha_hi].

    LOUD failure when even the smallest rung is degenerate: there is no
    coherent range to pick from, and a silent tiny alpha would fake a
    null result.
    """
    # Lazy import: eval.capability is the metrics home (pairs.py precedent
    # for extraction->eval imports).
    from ..eval.capability import distinct_ngram_ratio, perplexity
    from ..generate import generate_batch

    ladder = sorted(float(a) for a in alpha_ladder)
    if not ladder or ladder[0] <= 0:
        raise ValueError(f"alpha_ladder must be positive, got {alpha_ladder}")

    base_texts = generate_batch(
        model, tok, rendered_prompts, batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    base_gen_ppl = perplexity(model, tok, texts=[t for t in base_texts if t])

    sweep = []
    alpha_hi = None
    bracketed = False
    for i, alpha in enumerate(ladder):
        steered_texts = generate_batch(
            model, tok, rendered_prompts,
            sv={layer: direction}, scale=alpha, positions="last",
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        rep = distinct_ngram_ratio(steered_texts)
        gen_ppl = perplexity(model, tok, texts=[t for t in steered_texts if t])
        degenerate = bool(
            rep < rep_threshold or gen_ppl > ppl_guard_factor * base_gen_ppl
        )
        rung = {
            "alpha": alpha,
            "distinct3": rep,
            "gen_ppl": gen_ppl,
            "degenerate": degenerate,
        }
        sweep.append(rung)
        if log_fn is not None:
            log_fn({f"antidote_calib/{k}": v for k, v in rung.items()}, step=i)
        print(f"[antidote] alpha={alpha:g}: distinct3={rep:.3f} "
              f"gen_ppl={gen_ppl:.2f} (base {base_gen_ppl:.2f}) "
              f"{'DEGENERATE' if degenerate else 'coherent'}")
        if degenerate:
            bracketed = True
            break  # the first degenerate rung ends the coherent range
        alpha_hi = alpha

    if alpha_hi is None:
        raise RuntimeError(
            f"antidote calibration found NO coherent alpha: the smallest "
            f"ladder rung ({ladder[0]:g}) already fails the degeneration "
            f"guard (distinct3 < {rep_threshold} or gen_ppl > "
            f"{ppl_guard_factor}x base). Sweep: {sweep}"
        )
    warnings = []
    if not bracketed:
        warnings.append(
            f"incoherence bound not bracketed: every ladder rung up to "
            f"{ladder[-1]:g} stayed coherent — alpha_hi is a ladder "
            "ceiling, not a measured transition"
        )
        print(f"[antidote] WARNING: {warnings[-1]}")
    return {
        "base_alpha": alpha_hi / 2.0,  # midpoint of the coherent [0, hi]
        "alpha_bounds": [0.0, alpha_hi],
        "alpha_ladder": ladder,
        "rep_threshold": rep_threshold,
        "ppl_guard_factor": ppl_guard_factor,
        "base_gen_ppl": base_gen_ppl,
        "calib_max_new_tokens": max_new_tokens,
        "n_calib_prompts": len(rendered_prompts),
        "sweep": sweep,
        "warnings": warnings,
    }


class AntidoteObject:
    """Trained Antidote artifact behind the stage-2 object protocol.

    apply() steers at positions="last" with the ABSOLUTE alpha
    base_alpha * scale — the eval factor grid multiplies the stored
    base_alpha (metadata), never a norm-relative quantity. skip_bos is
    deliberately off: the paper has no BOS carve-out and the last-token
    mask cannot land on BOS for any real prompt.
    """

    kind = "antidote"

    def __init__(self, sv: SteeringVector):
        if sv.method != "antidote":
            raise ValueError(
                f"AntidoteObject requires a SteeringVector with "
                f"method='antidote', got {sv.method!r}"
            )
        if "base_alpha" not in sv.metadata:
            raise ValueError(
                "antidote artifact has no base_alpha in its metadata — "
                "an uncalibrated object cannot be applied (the factor "
                "grid multiplies base_alpha); retrain the cell"
            )
        self.sv = sv
        self.base_alpha = float(sv.metadata["base_alpha"])

    @property
    def trainable(self) -> list:
        return []  # closed-form extraction; nothing trains by gradient

    def n_params(self) -> int:
        return sum(v.numel() for v in self.sv.vectors.values())

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def apply(self, model, scale: float = 1.0):
        return steering(
            model, self.sv, scale=self.base_alpha * scale,
            positions="last", skip_bos=False, norm_relative=False,
        )

    def to_(self, device) -> None:
        self.sv.vectors = {
            l: v.to(device) for l, v in self.sv.vectors.items()
        }

    def save(self, path) -> None:
        self.sv.save(path)


def train_antidote(
    model,
    tok,
    instruction,
    pairs,
    *,
    layer: int,
    log_fn=None,
    k_frac: float = 0.05,
    alpha_ladder=DEFAULT_ALPHA_LADDER,
    calib_prompts: int = 16,
    calib_max_new_tokens: int = 128,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    batch_size: int = 8,
    epochs: int | None = None,   # accepted for train_kwargs.default compat;
    max_len: int | None = None,  # unused — the extraction is closed-form
) -> AntidoteObject:
    """The registry trainer: capture -> PCA -> sign -> mask -> calibrate.

    Contrast (SI steering adaptation): the SAME pair prompts rendered
    WITH the instruction as system prompt (target cluster) vs with NO
    system prompt (other cluster) — last PROMPT token, pre-generation,
    at the single pinned layer. epochs/max_len are accepted only because
    the shared train_kwargs.default carries them for the gradient arms.
    """
    from ..models import build_chat_prompt

    del epochs, max_len  # closed-form: documented no-ops (see docstring)
    prompts = [p.prompt for p in pairs]
    if not prompts:
        raise ValueError("antidote extraction got an empty pairs list")

    target_texts = [
        build_chat_prompt(tok, p, system=instruction.text) for p in prompts
    ]
    other_texts = [build_chat_prompt(tok, p, system=None) for p in prompts]
    acts_t = collect_pooled_activations(
        model, tok, target_texts, [layer], position="last",
        batch_size=batch_size,
    ).acts[layer]
    acts_o = collect_pooled_activations(
        model, tok, other_texts, [layer], position="last",
        batch_size=batch_size,
    ).acts[layer]

    # n-vs-d floor (2026-08-20): the PCA direction is a d-vector estimated
    # from the pooled states — n < d is legal (SVD of a thin matrix) but
    # must be labelled with the realised numbers, never silent.
    floor_info = {"sample_floor": [check_sample_floor(
        int(acts_t.shape[0] + acts_o.shape[0]), int(acts_t.shape[1]),
        what="pooled PCA rows", tag="antidote", log_fn=log_fn)]}
    pc1, dir_info = antidote_direction(acts_t, acts_o)
    masked, mask_info = sparsify(pc1, k_frac)

    calib = calibrate_base_alpha(
        model, tok, other_texts[:calib_prompts], masked, layer,
        alpha_ladder=alpha_ladder, rep_threshold=rep_threshold,
        ppl_guard_factor=ppl_guard_factor,
        max_new_tokens=calib_max_new_tokens, batch_size=batch_size,
        log_fn=log_fn,
    )

    metadata = {
        "paper": PAPER,
        "fidelity": (
            "paper-equation implementation, diffed against the authors' "
            "panda-guard RepeDefender (see extraction/antidote.py "
            "docstring for the documented divergences)"
        ),
        "contrast": "si_prompted_vs_unprompted_prompt_last_token",
        "positions": "last",
        "layer": layer,
        "n_pairs": len(prompts),
        **floor_info,
        **dir_info,
        **mask_info,
        **{k: v for k, v in calib.items() if k != "warnings"},
        "warnings": dir_info["warnings"] + calib["warnings"],
    }
    sv = SteeringVector(
        vectors={layer: masked},
        method="antidote",
        model_name=getattr(model.config, "_name_or_path", ""),
        instruction_id=instruction.id,
        metadata=metadata,
    )
    return AntidoteObject(sv)


# ---------------------------------------------------------------------------
# SI-AGNOSTIC fit-once mode (content-based; the paper-faithful setting)
# ---------------------------------------------------------------------------
# train_antidote (above) is the SI steering ADAPTATION: contrast = SI-present
# vs SI-absent renderings of the same prompts, one object per SI. The paper
# (panda-guard RepeDefender) contrasts HARMFUL vs BENIGN content and fits
# ONCE. train_antidote_content is that path (sets from data/content_sets.py)
# — and is strictly MORE faithful to the paper than the SI-mode contrast.


def train_antidote_content(
    model,
    tok,
    benign_prompts,
    harmful_prompts,
    *,
    layer: int,
    log_fn=None,
    k_frac: float = 0.05,
    alpha_ladder=DEFAULT_ALPHA_LADDER,
    calib_prompts: int = 16,
    calib_max_new_tokens: int = 128,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    batch_size: int = 8,
) -> AntidoteObject:
    """Fit Antidote ONCE on labeled harmful/benign CONTENT, SI-agnostically
    (panda-guard RepeDefender's own setting). The PCA safety direction is
    signed toward the HARMFUL cluster (target=harmful, other=benign) so
    +alpha adds refusal pressure; top-k% |component| mask; calibrated on the
    benign set (the deployed surface a defense must not over-refuse). Both
    sets are rendered SI-FREE (system=None), so the object carries no SI.

    The sign convention differs from the SI-mode trainer BY DESIGN: there
    target=SI-present (move toward SI-satisfaction); here target=harmful
    (move toward refusal), which is the jailbreak-defense direction."""
    from ..models import build_chat_prompt

    if not benign_prompts or not harmful_prompts:
        raise ValueError("antidote content fit needs non-empty benign and "
                         "harmful prompt lists")
    harmful_texts = [build_chat_prompt(tok, p, system=None)
                     for p in harmful_prompts]
    benign_texts = [build_chat_prompt(tok, p, system=None)
                    for p in benign_prompts]
    acts_harmful = collect_pooled_activations(
        model, tok, harmful_texts, [layer], position="last",
        batch_size=batch_size).acts[layer]
    acts_benign = collect_pooled_activations(
        model, tok, benign_texts, [layer], position="last",
        batch_size=batch_size).acts[layer]

    # n-vs-d floor (2026-08-20): label an n < d PCA fit with the realised
    # numbers (the 2,041-row default once fed a d = 4096 direction).
    floor_info = {"sample_floor": [check_sample_floor(
        int(acts_harmful.shape[0] + acts_benign.shape[0]),
        int(acts_harmful.shape[1]),
        what="pooled PCA rows", tag="antidote_content", log_fn=log_fn)]}
    pc1, dir_info = antidote_direction(acts_harmful, acts_benign)  # +→harmful
    masked, mask_info = sparsify(pc1, k_frac)

    calib = calibrate_base_alpha(
        model, tok, benign_texts[:calib_prompts], masked, layer,
        alpha_ladder=alpha_ladder, rep_threshold=rep_threshold,
        ppl_guard_factor=ppl_guard_factor,
        max_new_tokens=calib_max_new_tokens, batch_size=batch_size,
        log_fn=log_fn,
    )

    metadata = {
        "paper": PAPER,
        "fidelity": (
            "paper-equation implementation (panda-guard RepeDefender), "
            "harmful-vs-benign content contrast — the paper's own setting"
        ),
        "fit_mode": "content_si_agnostic",
        "contrast": "harmful_vs_benign_content_prompt_last_token",
        "positions": "last",
        "layer": layer,
        "n_benign": len(benign_prompts),
        "n_harmful": len(harmful_prompts),
        **floor_info,
        **dir_info,
        **mask_info,
        **{k: v for k, v in calib.items() if k != "warnings"},
        "warnings": dir_info["warnings"] + calib["warnings"],
    }
    sv = SteeringVector(
        vectors={layer: masked},
        method="antidote",
        model_name=getattr(model.config, "_name_or_path", ""),
        instruction_id="",   # SI-agnostic: no instruction
        metadata=metadata,
    )
    return AntidoteObject(sv)
