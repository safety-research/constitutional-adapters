"""AdaSteer (arXiv 2504.09466, EMNLP 2025) as an SI steering arm.

FIDELITY SOURCE: the paper LaTeX + the authors' code
github.com/MuyuenLP/AdaSteer @ 2a02b78 (verified 2026-08-17). AdaSteer
steers with TWO difference-in-means directions and PER-INPUT coefficients
fit from the input's projections onto them (the R-Law / H-Law):

- Rejection Direction (RD): a behavior/compliance axis.
- Harmfulness Direction (HD): an input-content axis; the applied compliance
  vector is the PROJECTION of HD onto RD (paper Eq. 9), so v_HD is COLLINEAR
  with RD per layer — HD's role is its per-input PROJECTION, which modulates
  the steering strength, not a second direction.
- position feature (the part the paper's equations omit, recovered from the
  code): a normalized affine coordinate
  pos = 100·(h − μ_a)·(μ_b − μ_a)/‖μ_b − μ_a‖², so μ_a ↦ 0 and μ_b ↦ +100.
- inference: h += α(x)·v_RD + β(x)·v_HD, added over the prompt positions on
  the PREFILL only; α, β come from the input's clean pos_RD / pos_HD.

DOCUMENTED DIVERGENCE (paper/code vs our call): the released code fits
R-Law/H-Law as a CLIPPED LINEAR map λ = w·pos + b (no sigmoid), then grid-
searches (w, b) on a validation set. The task specifies the method's
identity as LOGISTIC regression on the projections (with an AUC diagnostic),
which the "R-Law/H-Law" naming implies and which yields a bounded per-input
coefficient in [0, 1] naturally. We therefore fit a LOGISTIC regression on
each projection (pure-torch, no new dependency) and record its AUC; the
per-input coefficient is base_scale · P(needs-steering | pos). This is a
deliberate, documented adaptation of the fit form, not the released code.

DOCUMENTED DIVERGENCE (layers): the paper steers all layers with a separate
RD probe layer (8) and HD probe layer (13) for Llama-3.1. We pin the SINGLE
campaign layer 15 (block-15 output) — probe layer == steer layer — for
comparability with every other arm, which also collapses the paper's
two-pass (clean probe → steered) into one pass: at layer 15 the prefill's
last-token residual is already clean (nothing is steered before layer 15),
so pos_RD/pos_HD are read from it and the edit added in the same forward.

SI STEERING ADAPTATION (the data mapping):
- RD → the SI-compliance direction: mu(SI-present) − mu(SI-absent) last-token
  activations of the standard A1 pair prompts. +RD ⇒ toward SI-compliance.
  R-Law labels are the KNOWN capture condition — SI-present = satisfied (0),
  SI-absent = needs-steering (1) — the SI-present/absent split IS the
  compliance proxy here, so no teacher generation is needed for the fit.
- HD → the "SI-relevant input" direction: mu(benign) − mu(jb) over UNPROMPTED
  renderings (benign = A1 prompts, jb = HarmBench VAL behaviours). H-Law
  labels: benign = SI-relevant format target (1), jb = less (0). β then
  steers benign inputs toward format compliance more than adversarial ones —
  a documented sign choice (the paper concedes HD's DIRECTION is immaterial;
  only its per-input projection matters).
- base_scale (the absolute magnitude the paper leaves to model-specific
  tuning) is calibrated on the canary (distinct-3 + 3x-gen-ppl); the eval
  factor grid multiplies it.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models import build_chat_prompt
from .calibrate import DEFAULT_SCALE_LADDER, calibrate_base_scale
from .capture import collect_pooled_activations

PAPER = "arXiv:2504.09466 (AdaSteer, EMNLP 2025)"


def fit_logistic_1d(
    x: torch.Tensor, y: torch.Tensor, *, l2: float = 1e-2,
    iters: int = 300, lr: float = 0.1,
) -> dict:
    """1-D logistic regression P(y=1) = sigmoid(w·x + b), fit in pure torch
    (no sklearn dependency). Standardizes x internally for a stable fit and
    returns w, b in the ORIGINAL feature units plus the training AUC.

    A single-class y (degenerate) cannot be separated: returns w=0, b =
    logit(mean), AUC=0.5, and a warning — the caller records it rather than
    crashing on a tiny/degenerate fit set.
    """
    x = x.float().flatten()
    y = y.float().flatten()
    n = x.numel()
    frac = float(y.mean()) if n else 0.0
    if n < 2 or frac in (0.0, 1.0):
        p = min(max(frac, 1e-4), 1 - 1e-4)
        return {"w": 0.0, "b": float(torch.logit(torch.tensor(p))),
                "auc": 0.5, "n": int(n),
                "warning": "degenerate (single-class or <2 samples) — "
                           "constant coefficient"}
    mu, sd = x.mean(), x.std() + 1e-6
    xs = (x - mu) / sd
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(iters):
        loss = F.binary_cross_entropy_with_logits(w * xs + b, y) + l2 * w.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w_std = float(w.detach())
    b_std = float(b.detach())
    # Undo standardization: w·(x−mu)/sd + b = (w/sd)·x + (b − w·mu/sd).
    w_raw = w_std / float(sd)
    b_raw = b_std - w_std * float(mu) / float(sd)
    scores = torch.sigmoid(w_raw * x + b_raw)
    return {"w": w_raw, "b": b_raw, "auc": _auc(scores, y), "n": int(n),
            "warning": ""}


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC = P(score of a random positive > score of a random negative),
    computed by rank (ties count as 0.5)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    wins = (diff > 0).float().sum() + 0.5 * (diff == 0).float().sum()
    return float(wins / (pos.numel() * neg.numel()))


def affine_pos(h: torch.Tensor, mu_a: torch.Tensor, mu_b: torch.Tensor
               ) -> torch.Tensor:
    """AdaSteer's normalized position coordinate: 100·(h−μ_a)·(μ_b−μ_a)/
    ‖μ_b−μ_a‖² (μ_a ↦ 0, μ_b ↦ +100). h may be [d] or [B, d]."""
    delta = (mu_b - mu_a).float()
    denom = float(delta @ delta) + 1e-8
    return 100.0 * ((h.float() - mu_a.float()) @ delta) / denom


class AdaSteerObject:
    """Trained AdaSteer artifact behind the stage-2 object protocol.

    apply() registers a resid_post hook at the pinned layer that, during
    PREFILL ONLY, reads the clean last-token residual, computes the per-input
    coefficients from its pos_RD/pos_HD (logistic R-Law/H-Law), and adds
    base_scale·scale·(α·v_RD + β·v_HD) to every prompt position. Decode steps
    are untouched (prefill-only, per the paper)."""

    kind = "adasteer"

    def __init__(self, model, layer: int, params: dict,
                 base_scale: float, metadata: dict | None = None):
        from ..models import get_decoder_layers

        self.layer = layer
        self.base_scale = float(base_scale)
        self.metadata = metadata or {}
        self.p = {}  # tensors moved to device by to_
        for key in ("v_RD", "v_HD", "mu_a_rd", "mu_b_rd", "mu_a_hd", "mu_b_hd"):
            self.p[key] = params[key].float()
        self.rlaw = dict(params["rlaw"])   # {w, b}
        self.hlaw = dict(params["hlaw"])
        self._decoder = get_decoder_layers(model)

    @property
    def trainable(self) -> list:
        return []

    def n_params(self) -> int:
        return int(self.p["v_RD"].numel() + self.p["v_HD"].numel())

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def _coeffs(self, h_last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row α, β [B] from the clean last-token residual."""
        pos_rd = affine_pos(h_last, self.p["mu_a_rd"], self.p["mu_b_rd"])
        pos_hd = affine_pos(h_last, self.p["mu_a_hd"], self.p["mu_b_hd"])
        alpha = torch.sigmoid(self.rlaw["w"] * pos_rd + self.rlaw["b"])
        beta = torch.sigmoid(self.hlaw["w"] * pos_hd + self.hlaw["b"])
        return alpha, beta

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mag = self.base_scale * scale

        def hook(_mod, _args, output):
            if mag == 0.0:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] <= 1:
                return output  # decode step: prefill-only
            dev, dt = hidden.device, hidden.dtype
            v_rd = self.p["v_RD"].to(device=dev)
            v_hd = self.p["v_HD"].to(device=dev)
            h_last = hidden[:, -1, :].float()      # [B, d] clean last token
            alpha, beta = self._coeffs(h_last)     # [B], [B]
            edit = (alpha.unsqueeze(1) * v_rd + beta.unsqueeze(1) * v_hd)  # [B, d]
            add = (mag * edit).unsqueeze(1).to(dt)  # broadcast to all positions
            steered = hidden + add
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        h = self._decoder[self.layer].register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()

    def to_(self, device) -> None:
        self.p = {k: v.to(device) for k, v in self.p.items()}

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind, "layer": self.layer,
             "base_scale": self.base_scale,
             "params": {k: v.detach().float().cpu() for k, v in self.p.items()},
             "rlaw": self.rlaw, "hlaw": self.hlaw,
             "metadata": self.metadata},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "AdaSteerObject":
        params = dict(blob["params"])
        params["rlaw"] = blob["rlaw"]
        params["hlaw"] = blob["hlaw"]
        return cls(model, layer=int(blob["layer"]), params=params,
                   base_scale=float(blob["base_scale"]),
                   metadata=blob.get("metadata", {}))


def build_directions(benign_si, benign_no, jb_no) -> dict:
    """Closed-form RD/HD directions + centers + collinear v_HD from three
    pooled activation sets ([N, d] each): benign SI-present, benign SI-absent,
    jb SI-absent. Pure arithmetic — the unit-testable core."""
    mu_bs = benign_si.float().mean(0)
    mu_bn = benign_no.float().mean(0)
    mu_jn = jb_no.float().mean(0)
    rd = mu_bs - mu_bn                       # SI direction (toward compliance)
    hd = mu_bn - mu_jn                       # content axis (benign − jb)
    c = float(rd @ hd) / (float(rd @ rd) + 1e-8)
    v_hd = c * rd                            # Eq. 9 projection ⇒ collinear
    return {
        "v_RD": rd, "v_HD": v_hd,
        "mu_a_rd": mu_bn, "mu_b_rd": mu_bs,   # SI-absent ↦ 0, SI-present ↦ 100
        "mu_a_hd": mu_jn, "mu_b_hd": mu_bn,   # jb ↦ 0, benign ↦ 100
        "collinearity_c": c,
        "rd_norm": float(rd.norm()), "hd_norm": float(hd.norm()),
    }


def train_adasteer(
    model,
    tok,
    instruction,
    pairs,
    *,
    layer: int,
    log_fn=None,
    jb_prompts: list[str] | None = None,
    jb_split: str = "val",
    n_jb: int | None = None,
    batch_size: int = 8,
    calib_prompts: int = 16,
    calib_max_new_tokens: int = 128,
    scale_ladder=DEFAULT_SCALE_LADDER,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 3.0,
    epochs: int | None = None,  # train_kwargs.default compat (documented no-op)
    max_len: int | None = None,  # train_kwargs.default compat (documented no-op)
) -> AdaSteerObject:
    """Registry trainer: capture the three last-token activation sets, build
    RD/HD + collinear v_HD, fit the logistic R-Law/H-Law on the projections,
    and calibrate the base scale on the canary procedure. ``jb_prompts`` is
    the test seam (production loads HarmBench VAL). epochs/max_len are
    documented no-ops (closed-form + logistic, no gradient LM training)."""
    del epochs, max_len
    prompts = [p.prompt for p in pairs]
    if not prompts:
        raise ValueError("AdaSteer extraction got an empty pairs list")
    if jb_prompts is None:
        from ..eval.jailbreak_data import load_harmbench

        jb_prompts = [g.text for g in load_harmbench(split=jb_split)]
    if not jb_prompts:
        raise RuntimeError(
            f"AdaSteer found no jb prompts (HarmBench split={jb_split!r}) for "
            "the HD content axis")
    if n_jb is not None:
        jb_prompts = jb_prompts[:n_jb]

    benign_si_txt = [build_chat_prompt(tok, p, system=instruction.text)
                     for p in prompts]
    benign_no_txt = [build_chat_prompt(tok, p, system=None) for p in prompts]
    jb_no_txt = [build_chat_prompt(tok, p, system=None) for p in jb_prompts]

    def _cap(texts):
        return collect_pooled_activations(
            model, tok, texts, [layer], position="last",
            batch_size=batch_size).acts[layer]

    a_bs = _cap(benign_si_txt)
    a_bn = _cap(benign_no_txt)
    a_jn = _cap(jb_no_txt)

    dirs = build_directions(a_bs, a_bn, a_jn)

    # R-Law: pos_RD of {SI-present (label 0, satisfied), SI-absent (label 1,
    # needs steering)}; logistic separates them by the SI-in-context coord.
    pos_rd = torch.cat([
        affine_pos(a_bs, dirs["mu_a_rd"], dirs["mu_b_rd"]),
        affine_pos(a_bn, dirs["mu_a_rd"], dirs["mu_b_rd"]),
    ])
    y_rd = torch.cat([torch.zeros(len(a_bs)), torch.ones(len(a_bn))])
    rlaw = fit_logistic_1d(pos_rd, y_rd)

    # H-Law: pos_HD of {benign SI-absent (label 1, relevant), jb SI-absent
    # (label 0)}; logistic separates benign from adversarial content.
    pos_hd = torch.cat([
        affine_pos(a_bn, dirs["mu_a_hd"], dirs["mu_b_hd"]),
        affine_pos(a_jn, dirs["mu_a_hd"], dirs["mu_b_hd"]),
    ])
    y_hd = torch.cat([torch.ones(len(a_bn)), torch.zeros(len(a_jn))])
    hlaw = fit_logistic_1d(pos_hd, y_hd)

    params = {**{k: dirs[k] for k in
                 ("v_RD", "v_HD", "mu_a_rd", "mu_b_rd", "mu_a_hd", "mu_b_hd")},
              "rlaw": {"w": rlaw["w"], "b": rlaw["b"]},
              "hlaw": {"w": hlaw["w"], "b": hlaw["b"]}}
    obj = AdaSteerObject(model, layer, params, base_scale=1.0)
    obj.to_(next(model.parameters()).device)

    calib_rendered = benign_no_txt[:calib_prompts]

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
        log_fn=log_fn, arm="adasteer",
    )
    obj.base_scale = calib["base_scale"]
    warnings = [w for w in (rlaw["warning"], hlaw["warning"]) if w] \
        + calib["warnings"]
    obj.metadata = {
        "paper": PAPER,
        "layer": layer,
        "n_benign": len(prompts),
        "n_jb": len(jb_prompts),
        "collinearity_c": dirs["collinearity_c"],
        "rd_norm": dirs["rd_norm"], "hd_norm": dirs["hd_norm"],
        "rlaw_auc": rlaw["auc"], "hlaw_auc": hlaw["auc"],
        "rlaw_w": rlaw["w"], "rlaw_b": rlaw["b"],
        "hlaw_w": hlaw["w"], "hlaw_b": hlaw["b"],
        **{k: v for k, v in calib.items() if k != "warnings"},
        "warnings": warnings,
    }
    return obj
