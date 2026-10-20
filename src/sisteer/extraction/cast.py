"""CAST — Conditional Activation Steering (Lee et al., arXiv 2409.05907,
ICLR 2025 spotlight) as a prompt-compiling (Class A) two-SI arm.

FIDELITY SOURCE: the authors' original library, github.com/IBM/activation-
steering @ 52be60235ee309b46c49d6d5877f36e20c52e6ab (verified 2026-08-18
against the arXiv HTML). CAST is a HYBRID of a behavior vector and a
CONDITION: the behavior vector fires only when a rank-1 condition on the
prompt's activations is met.

Paper mechanism (Section 3), reproduced here:
    proj_c(h) = (c cᵀ / c·c) h              # rank-1 projection onto c
    sim(h)    = cos(h, tanh(proj_c(h)))     # tanh for stability (paper §3)
    fire      = sim  ⧀ θ                     # ⧀ per the searched direction
    h' = h + fire · α · v̂                    # v̂ unit behavior direction

Faithful specifics carried from the original code:
- BOTH vectors are the first PCA component of CONTRASTIVE activations
  (steering_vector.py read_representations), NOT plain diff-in-means. The
  paper-reproducing variant is ``pca_pairwise``: subtract each pair's
  midpoint from both members, then PC1 (unit norm), sign-fixed so the
  POSITIVE cluster projects larger. layer i = block i output (the harness
  convention; the original's hidden_states[i+1]).
- The behavior vector pools the CONTRAST completion tokens; the condition
  vector pools the whole prompt (the original's "suffix-only" vs "all").
- The (condition-layer, threshold, direction) triple is chosen by an F1
  grid search over labeled should-fire / should-not-fire prompts
  (malleable_model.find_best_condition_point). The direction naming is the
  original's INVERTED convention: 'smaller' ⇒ fire when sim > θ, 'larger'
  ⇒ fire when sim < θ.
- The condition is evaluated ONCE, on the prompt's prefill pass, and
  LATCHED for the rest of generation (leash_layer._process_single_
  condition, gated on forward_calls == 1). If it fires, the behavior
  vector is added to every position from that pass onward.

SI STEERING ADAPTATIONS (deliberate, mirroring every other arm):
- The behavior vector is SI-DERIVED (Class A): the contrast is the SI-
  prompted completion (pos) vs the no-SI completion (neg) under the same
  user prompt — the same pos/neg the distillation arms consume — pooled
  over completion tokens. This is what makes CAST one object PER SI.
- The condition is CATEGORY-based and SI-AGNOSTIC: harmful (should-fire)
  vs benign (should-not-fire) PROMPTS, supplied explicitly as
  condition_pos_prompts / condition_neg_prompts. If they are absent the
  trainer REFUSES unless allow_pairs_fallback=True, in which case it
  derives a degenerate SI-present-vs-SI-absent condition from the pairs
  and RECORDS fit_path="pairs_fallback_si_derived" in the metadata (a
  loud, auditable fallback — an SI-derived condition is not the paper's
  category condition and must never pass silently).
- The unit behavior direction is tiny against the residual norm, so a base
  scale is calibrated on the FIRE path (the shared incoherence-ladder
  guard, as antidote/alphasteer do); the eval factor grid multiplies it.

BATCHING/PADDING: the original assumes batch size 1 (hidden_states[0]).
This implementation latches PER ROW from the 2D attention mask so the
conditional gate is correct under generate.py's forced left padding, and
the reset keys on the prefill event (seq_len > 1) so one apply() spanning
several generate calls stays exact.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from ..models import build_chat_prompt, get_decoder_layers, num_layers
from .calibrate import DEFAULT_SCALE_LADDER, calibrate_base_scale
from .capture import collect_pooled_activations

PAPER = "arXiv:2409.05907 (Conditional Activation Steering, Lee et al., ICLR 2025)"
PCA_METHODS = ("pca_pairwise", "pca_center", "pca_diff")


# -- direction extraction (steering_vector.read_representations) ----------------


def pca_direction(
    pos: torch.Tensor, neg: torch.Tensor, method: str = "pca_pairwise"
) -> tuple[torch.Tensor, dict]:
    """First PCA component of the contrastive activations, sign-fixed so the
    POSITIVE cluster projects larger. pos/neg: [N, d] pooled activations.

    Faithful to the original read_representations (steering_vector.py:184):
    - pca_pairwise (paper-reproducing): center each pair at its midpoint,
      then PC1 over the stacked ±deviations;
    - pca_center: PC1 over all activations centered at the global mean;
    - pca_diff: PC1 over the pairwise differences pos−neg.
    sklearn PCA(n_components=1) == the top right singular vector of the
    globally-centered training matrix; the component is unit norm.
    """
    if method not in PCA_METHODS:
        raise ValueError(f"method must be one of {PCA_METHODS}, got {method!r}")
    P, N = pos.float(), neg.float()
    if P.shape != N.shape:
        raise ValueError(
            f"pca_direction needs equal-shaped pos/neg, got {tuple(P.shape)} "
            f"vs {tuple(N.shape)}")
    if P.shape[0] < 1:
        raise ValueError("pca_direction got an empty contrast set")

    if method == "pca_diff":
        train = P - N
    elif method == "pca_center":
        train = torch.cat([P, N], dim=0)
        train = train - train.mean(dim=0)
    else:  # pca_pairwise
        center = (P + N) / 2.0
        train = torch.cat([P - center, N - center], dim=0)

    Xc = train - train.mean(dim=0)  # sklearn PCA centers before the SVD
    if Xc.shape[0] < 2:
        # A single contrast row: PC1 is just that (unit) direction.
        pc1 = Xc[0] / (Xc[0].norm() + 1e-8)
        evr = 1.0
    else:
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        pc1 = Vh[0]
        total = float((S**2).sum())
        evr = float(S[0] ** 2) / total if total > 0 else 0.0
    pc1 = pc1 / (pc1.norm() + 1e-8)

    proj_pos = float((P @ pc1).mean())
    proj_neg = float((N @ pc1).mean())
    flipped = proj_pos < proj_neg
    if flipped:
        pc1 = -pc1
    info = {
        "pca_method": method,
        "explained_variance_ratio_pc1": evr,
        "sign_flipped": bool(flipped),
        "proj_gap": abs(proj_pos - proj_neg),
    }
    return pc1, info


# -- condition similarity (leash_layer._process_single_condition) ---------------


def condition_similarity(
    pooled: torch.Tensor, condition_vec: torch.Tensor
) -> torch.Tensor:
    """cos(h, tanh(P·h)), P = c cᵀ / (c·c), the exact CAST condition score.

    pooled: [..., d] one pooled hidden per example; condition_vec: [d].
    Returns the per-example similarity (shape pooled.shape[:-1]). The tanh
    is the paper's stability transform (§3); without it this reduces to
    |cos(h, c)|.
    """
    c = condition_vec.float()
    h = pooled.float()
    cc = float(c @ c) + 1e-8
    coeff = (h @ c) / cc                      # [...]
    proj = coeff.unsqueeze(-1) * c            # [..., d]  == P·h
    tanh_proj = torch.tanh(proj)
    num = (h * tanh_proj).sum(dim=-1)
    den = h.norm(dim=-1) * tanh_proj.norm(dim=-1) + 1e-8
    return num / den


def _fires(sim: torch.Tensor, threshold: float, direction: str) -> torch.Tensor:
    """The original's INVERTED naming (leash_layer.py): 'smaller' ⇒ fire when
    sim > θ; 'larger' ⇒ fire when sim < θ."""
    if direction == "smaller":
        return sim > threshold
    if direction == "larger":
        return sim < threshold
    raise ValueError(f"direction must be 'smaller' or 'larger', got {direction!r}")


def search_condition_point(
    pos_sims: dict[int, torch.Tensor],
    neg_sims: dict[int, torch.Tensor],
    *,
    threshold_range: tuple[float, float] = (-1.0, 1.0),
    threshold_step: float = 0.02,
) -> dict:
    """F1 grid search over (condition layer, threshold, direction), maximizing
    separation of should-fire (pos) vs should-not-fire (neg) prompts —
    faithful to malleable_model.find_best_condition_point (single condition
    layer, max_layers_to_combine=1). pos_sims/neg_sims: {layer: [N] sims}.
    """
    lo, hi = threshold_range
    n_steps = max(1, int(round((hi - lo) / threshold_step)))
    thresholds = [lo + i * threshold_step for i in range(n_steps + 1)]
    best = {"f1": -1.0}
    for layer in sorted(pos_sims):
        ps, ns = pos_sims[layer], neg_sims[layer]
        for thr in thresholds:
            for direction in ("smaller", "larger"):
                tp = int(_fires(ps, thr, direction).sum())
                fp = int(_fires(ns, thr, direction).sum())
                fn = len(ps) - tp
                denom = 2 * tp + fp + fn
                f1 = (2 * tp / denom) if denom > 0 else 0.0
                if f1 > best["f1"]:
                    best = {
                        "condition_layer": int(layer),
                        "threshold": round(float(thr), 4),
                        "direction": direction,
                        "f1": float(f1),
                        "tp": tp, "fp": fp, "fn": fn,
                        "n_pos": len(ps), "n_neg": len(ns),
                    }
    return best


# -- pooling helpers for the inference hook (batch/left-pad correct) ------------


def _pool_prefill(hidden: torch.Tensor, mask: torch.Tensor | None,
                  mode: str) -> torch.Tensor:
    """Pool a prefill hidden [B, S, d] into [B, d] over REAL tokens.

    mode="mean": masked mean over non-pad positions (the original's default,
    but pad-masked — the original pooled padded positions too, a known quirk
    we correct for batched generation). mode="last": the final real token
    (== index S-1 under generate.py's left padding)."""
    B, S, _ = hidden.shape
    if mask is None or tuple(mask.shape) != (B, S):
        if mode == "last":
            return hidden[:, -1, :]
        return hidden.mean(dim=1)
    m = mask.to(hidden.dtype)
    if mode == "last":
        # index of the last real (non-pad) token per row
        last = (S - 1 - m.flip(1).argmax(dim=1)).long()
        return hidden[torch.arange(B, device=hidden.device), last]
    denom = m.sum(dim=1, keepdim=True).clamp(min=1)
    return (hidden * m.unsqueeze(-1)).sum(dim=1) / denom


class CastObject:
    """Trained CAST artifact behind the stage-2 object protocol.

    apply() installs (a) a model pre-hook capturing the 2D attention mask,
    (b) a condition-layer hook that latches a per-row fire mask on the
    prefill, and (c) behavior-layer hooks that add base_scale·scale·v̂ to
    the fired rows. skip_bos is off (the paper has no BOS carve-out) and the
    behavior is added to all positions of a fired row, matching the original
    apply_behavior_on_first_call=True default.
    """

    kind = "cast"

    def __init__(self, model, behavior_layers: list[int],
                 behavior_vecs: dict[int, torch.Tensor], condition_layer: int,
                 condition_vec: torch.Tensor, threshold: float, direction: str,
                 base_scale: float, *, pool_mode: str = "mean",
                 metadata: dict | None = None):
        if condition_layer >= min(behavior_layers):
            raise ValueError(
                f"CAST requires the condition layer ({condition_layer}) to "
                f"precede every behavior layer ({sorted(behavior_layers)}) so "
                "the fire latch is set before the behavior applies within one "
                "prefill pass (leash_layer ordering)")
        self.behavior_layers = sorted(behavior_layers)
        self.behavior_vecs = {int(l): v.float() for l, v in behavior_vecs.items()}
        self.condition_layer = int(condition_layer)
        self.condition_vec = condition_vec.float()
        self.threshold = float(threshold)
        self.direction = direction
        self.base_scale = float(base_scale)
        self.pool_mode = pool_mode
        self.metadata = metadata or {}
        self._decoder = get_decoder_layers(model)

    @property
    def trainable(self) -> list:
        return []  # PCA + F1 search: nothing trains by gradient

    def n_params(self) -> int:
        return (sum(v.numel() for v in self.behavior_vecs.values())
                + self.condition_vec.numel())

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mag = self.base_scale * scale
        state: dict = {"fired": None, "mask": None}

        def capture_mask(_mod, args, kwargs):
            m = kwargs.get("attention_mask")
            state["mask"] = m if (m is not None and m.dim() == 2) else None
            return None

        def condition_hook(_mod, _args, output):
            if mag == 0.0:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] <= 1:
                return output  # decode step: the latch persists, never recomputed
            pooled = _pool_prefill(hidden, state["mask"], self.pool_mode)
            c = self.condition_vec.to(device=hidden.device, dtype=torch.float32)
            sim = condition_similarity(pooled.float(), c)      # [B]
            state["fired"] = _fires(sim, self.threshold, self.direction)
            return output

        def make_behavior_hook(layer: int):
            vec = self.behavior_vecs[layer]

            def behavior_hook(_mod, _args, output):
                if mag == 0.0:
                    return output
                fired = state["fired"]
                if fired is None:
                    return output  # condition never ran (unbatched 1-token?)
                hidden = output[0] if isinstance(output, tuple) else output
                v = vec.to(device=hidden.device, dtype=hidden.dtype)
                row = fired.to(device=hidden.device, dtype=hidden.dtype)
                add = (mag * v).view(1, 1, -1) * row.view(-1, 1, 1)
                steered = hidden + add
                if isinstance(output, tuple):
                    return (steered,) + output[1:]
                return steered

            return behavior_hook

        handles = [
            model.register_forward_pre_hook(capture_mask, with_kwargs=True),
            self._decoder[self.condition_layer].register_forward_hook(
                condition_hook),
        ]
        for l in self.behavior_layers:
            handles.append(
                self._decoder[l].register_forward_hook(make_behavior_hook(l)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def to_(self, device) -> None:
        self.behavior_vecs = {l: v.to(device) for l, v in self.behavior_vecs.items()}
        self.condition_vec = self.condition_vec.to(device)

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind,
             "behavior_layers": list(self.behavior_layers),
             "behavior_vecs": {str(l): v.detach().float().cpu()
                               for l, v in self.behavior_vecs.items()},
             "condition_layer": self.condition_layer,
             "condition_vec": self.condition_vec.detach().float().cpu(),
             "threshold": self.threshold,
             "direction": self.direction,
             "base_scale": self.base_scale,
             "pool_mode": self.pool_mode,
             "metadata": self.metadata},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "CastObject":
        return cls(
            model,
            behavior_layers=[int(l) for l in blob["behavior_layers"]],
            behavior_vecs={int(l): v for l, v in blob["behavior_vecs"].items()},
            condition_layer=int(blob["condition_layer"]),
            condition_vec=blob["condition_vec"],
            threshold=float(blob["threshold"]),
            direction=blob["direction"],
            base_scale=float(blob["base_scale"]),
            pool_mode=blob.get("pool_mode", "mean"),
            metadata=blob.get("metadata", {}),
        )


# -- default condition-layer candidate set --------------------------------------


def default_condition_layers(behavior_layers: list[int], n_model_layers: int
                             ) -> list[int]:
    """A handful of candidate condition layers strictly below the behavior
    layer (the original searches a lower range, e.g. (1, 10) for behavior
    15-23). Fractions of min(behavior_layer), clamped to [1, min_bl-1]."""
    min_bl = min(behavior_layers)
    cands = sorted({
        int(min_bl * r) for r in (0.2, 0.35, 0.5, 0.65, 0.8)
    } - {0})
    cands = [l for l in cands if 1 <= l < min_bl and l < n_model_layers]
    return cands or [max(1, min_bl - 1)]


def train_cast(
    model,
    tok,
    instruction,
    pairs,
    *,
    layer: int,
    log_fn=None,
    behavior_layers: list[int] | None = None,
    condition_layers: list[int] | None = None,
    condition_pos_prompts: list[str] | None = None,
    condition_neg_prompts: list[str] | None = None,
    allow_pairs_fallback: bool = False,
    method: str = "pca_pairwise",
    pool_mode: str = "mean",
    threshold_range: tuple[float, float] = (-1.0, 1.0),
    threshold_step: float = 0.02,
    batch_size: int = 8,
    calib_prompts: int = 16,
    calib_max_new_tokens: int = 128,
    scale_ladder=DEFAULT_SCALE_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    epochs: int | None = None,   # train_kwargs.default compat (documented no-op)
    max_len: int | None = None,  # train_kwargs.default compat (documented no-op)
) -> CastObject:
    """Registry trainer: SI-derived behavior vector (PCA over the pairs'
    SI-vs-no-SI completions) + a category-based condition (harmful vs benign
    prompts), the F1-searched fire rule, and a calibrated base scale.

    condition_pos_prompts (should-fire, e.g. harmful) / condition_neg_prompts
    (should-not-fire, e.g. benign) are the PRODUCTION path. Absent them, the
    trainer refuses unless allow_pairs_fallback=True — the fallback derives a
    degenerate SI-present-vs-SI-absent condition from the pairs and records
    fit_path="pairs_fallback_si_derived" (never silent: an SI-derived
    condition is not the paper's category condition). epochs/max_len are
    documented no-ops (the extraction is closed-form)."""
    del epochs, max_len
    prompts = [p.prompt for p in pairs]
    if not prompts:
        raise ValueError("CAST extraction got an empty pairs list")
    behavior_layers = behavior_layers or [layer]
    if condition_layers is None:
        condition_layers = default_condition_layers(
            behavior_layers, num_layers(model))

    # -- behavior vector: SI-prompted completion vs no-SI completion ----------
    # (the same pos/neg the distillation arms consume), pooled over completion
    # tokens, PCA per behavior layer. SI-derived => one object per SI.
    prompt_no_si = [build_chat_prompt(tok, p, system=None) for p in prompts]
    pos_texts, neg_texts, pos_lens, neg_lens = [], [], [], []
    for rp, pr in zip(prompt_no_si, pairs):
        n_prompt = len(tok(rp, add_special_tokens=True)["input_ids"])
        pos_texts.append(rp + pr.pos)
        neg_texts.append(rp + pr.neg)
        pos_lens.append(n_prompt)
        neg_lens.append(n_prompt)
    pos_acts = collect_pooled_activations(
        model, tok, pos_texts, behavior_layers, position="completion_mean",
        prompt_lens=pos_lens, batch_size=batch_size).acts
    neg_acts = collect_pooled_activations(
        model, tok, neg_texts, behavior_layers, position="completion_mean",
        prompt_lens=neg_lens, batch_size=batch_size).acts
    behavior_vecs: dict[int, torch.Tensor] = {}
    behavior_info: dict[int, dict] = {}
    for l in behavior_layers:
        v, info = pca_direction(pos_acts[l], neg_acts[l], method=method)
        behavior_vecs[l] = v
        behavior_info[l] = info

    # -- condition data: category-based (production) or pairs fallback --------
    if condition_pos_prompts is not None and condition_neg_prompts is not None:
        fit_path = "explicit_kwargs"
        cond_pos_rendered = [build_chat_prompt(tok, p, system=None)
                             for p in condition_pos_prompts]
        cond_neg_rendered = [build_chat_prompt(tok, p, system=None)
                             for p in condition_neg_prompts]
    else:
        if not allow_pairs_fallback:
            raise ValueError(
                "CAST needs condition_pos_prompts + condition_neg_prompts "
                "(the category-based should-fire / should-not-fire sets). "
                "None supplied and allow_pairs_fallback=False — refusing to "
                "silently substitute an SI-derived condition, which is not "
                "the paper's category condition. Pass the prompts, or set "
                "allow_pairs_fallback=True to accept the recorded fallback.")
        fit_path = "pairs_fallback_si_derived"
        cond_pos_rendered = [build_chat_prompt(tok, p, system=instruction.text)
                             for p in prompts]
        cond_neg_rendered = [build_chat_prompt(tok, p, system=None)
                             for p in prompts]

    # -- condition vectors per candidate layer + the F1 search ----------------
    cond_pos_acts = collect_pooled_activations(
        model, tok, cond_pos_rendered, condition_layers, position="mean",
        batch_size=batch_size).acts
    cond_neg_acts = collect_pooled_activations(
        model, tok, cond_neg_rendered, condition_layers, position="mean",
        batch_size=batch_size).acts
    condition_vecs: dict[int, torch.Tensor] = {}
    pos_sims: dict[int, torch.Tensor] = {}
    neg_sims: dict[int, torch.Tensor] = {}
    for l in condition_layers:
        cvec, _ = pca_direction(cond_pos_acts[l], cond_neg_acts[l], method=method)
        condition_vecs[l] = cvec
        pos_sims[l] = condition_similarity(cond_pos_acts[l], cvec)
        neg_sims[l] = condition_similarity(cond_neg_acts[l], cvec)
    best = search_condition_point(
        pos_sims, neg_sims, threshold_range=threshold_range,
        threshold_step=threshold_step)
    cond_layer = best["condition_layer"]
    if cond_layer >= min(behavior_layers):
        raise ValueError(
            f"searched condition layer {cond_layer} is not below the behavior "
            f"layer {min(behavior_layers)} — narrow condition_layers so the "
            "latch precedes the behavior (leash_layer ordering)")

    # -- calibrate the behavior scale on the (unconditional) FIRE path --------
    from ..generate import generate_batch

    calib_rendered = [build_chat_prompt(tok, p, system=None)
                      for p in prompts[:calib_prompts]]

    def _steered_generate(scale: float) -> list[str]:
        return generate_batch(
            model, tok, calib_rendered, sv=behavior_vecs, scale=scale,
            positions="all", batch_size=batch_size,
            max_new_tokens=calib_max_new_tokens)

    calib = calibrate_base_scale(
        model, tok, calib_rendered, _steered_generate,
        scale_ladder=scale_ladder, rep_threshold=rep_threshold,
        ppl_guard_factor=ppl_guard_factor,
        max_new_tokens=calib_max_new_tokens, batch_size=batch_size,
        log_fn=log_fn, arm="cast")

    metadata = {
        "paper": PAPER,
        "fit_path": fit_path,
        "pca_method": method,
        "behavior_layers": list(behavior_layers),
        "behavior_info": {str(l): behavior_info[l] for l in behavior_layers},
        "condition_layer": cond_layer,
        "condition_candidates": list(condition_layers),
        "threshold": best["threshold"],
        "direction": best["direction"],
        "condition_f1": best["f1"],
        "condition_confusion": {k: best[k] for k in ("tp", "fp", "fn")},
        "n_pairs": len(prompts),
        "n_condition_pos": len(cond_pos_rendered),
        "n_condition_neg": len(cond_neg_rendered),
        "pool_mode": pool_mode,
        **{k: v for k, v in calib.items() if k != "warnings"},
        "warnings": calib["warnings"],
    }
    obj = CastObject(
        model, behavior_layers, behavior_vecs, cond_layer, condition_vecs[cond_layer],
        best["threshold"], best["direction"], base_scale=calib["base_scale"],
        pool_mode=pool_mode, metadata=metadata)
    obj.to_(next(model.parameters()).device)
    return obj
