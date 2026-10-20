"""Circuit Breakers / Representation Rerouting (Zou et al., "Improving
Alignment and Robustness with Circuit Breakers", NeurIPS 2024, arXiv
2406.04313) as an SI-agnostic (Class B) two-SI baseline.

FIDELITY SOURCE: the authors' original code, github.com/GraySwanAI/
circuit-breakers @ a1ba1a5daced0288d126beda4cd900c92d48eb93 (verified
2026-08-18 against the arXiv HTML v4). The method trains a LoRA adapter so
that HARMFUL representations are "rerouted" (driven orthogonal to their
original direction) while BENIGN representations are retained.

Loss (src/lorra_circuit_breaker.py, paper Alg. 1), reproduced here:
    L = c_r · L_retain + c_s · L_reroute
    L_retain  = mean_tokens ‖ h_lora(x_retain) − h_orig(x_retain) ‖₂     (all layers)
    L_reroute = mean_tokens ReLU( cos( h_lora(x_harm), h_orig(x_harm) ) ) (target layers)
    c_r = α · progress,  c_s = α · (1 − progress),  progress = step / (2·max_steps)
Early training is reroute-dominated (c_s ≈ α), late training retain-
dominated (both → α/2). α = lorra_alpha = 10 for Llama-3-8B. h_orig is the
frozen base (adapters disabled); h_lora is the adapted forward — the
original runs one PEFT model with model.disable_adapter() for the
reference pass and the adapted pass for the gradient.

HARNESS ADAPTATION (deliberate — the frozen-base / one-object-per-cell
design):
- The original MERGES the trained LoRA into the weights and serves a plain
  model with NO runtime hook. Here the trained low-rank deltas live in a
  CircuitBreakerObject applied through forward hooks on the same seven
  projections (q/k/v/o/gate/up/down) over layers 0..max(target_layers),
  leaving the base frozen. This is behaviourally the merged adapter (the
  same additive dW = scaling·B·A·x per projection) but non-destructive and
  saveable as one object — so h_orig is simply the forward WITHOUT the
  object applied, and h_lora the forward WITH it, exactly mirroring
  disable_adapter().
- SI-AGNOSTIC (Class B, fit once): no instruction enters the signal. The
  retain set is benign prompt+completion, the reroute set is harmful
  prompt + HARMFUL CONTINUATION (the store's none-condition jailbroken
  assistant continuations — RR needs the continuation, not the bare
  prompt). Supplied via retain_texts / reroute_texts (already rendered).
  Absent them the trainer REFUSES unless allow_pairs_fallback=True, which
  derives retain=prompt+pos / reroute=prompt+neg from the pairs and records
  fit_path="pairs_fallback_si_derived" — a loud, auditable fallback (an
  SI-derived Class B object is a meaningless baseline if it passes
  silently).
- LoRA magnitude is LEARNED, so there is no closed-form scale to calibrate:
  base_scale is fixed at 1.0 (scale 1.0 = the trained adapter) and the eval
  factor grid multiplies it, like every other object.

DOCUMENTED DIVERGENCES from the original code:
1. RETAIN MASK. The original's retain mean is pad-DILUTED (nanmean over
   zeroed pad positions); we use a proper masked mean over real tokens
   (the reroute mean already is, in the original). A correctness fix for
   batched training, recorded here.
2. LAYER TRUNCATION. The original drops decoder layers above max(target)
   during training for speed and re-attaches them at merge. We keep the
   full model (the hook object never mutates it) and simply do not read or
   edit layers above max(target); the arithmetic on the touched layers is
   identical.
3. AUGMENTATION. The original's Llama-only 50%-prompt-drop and refusal-
   retain augmentations are dataset-construction choices left to the
   caller of retain_texts/reroute_texts, not baked into the trainer.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..models import build_chat_prompt, get_decoder_layers, num_layers

PAPER = "arXiv:2406.04313 (Circuit Breakers / Representation Rerouting, Zou et al., NeurIPS 2024)"
# The seven projections the original adapts (scripts/*.sh target_modules).
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")
DEFAULT_TARGET_LAYERS = (10, 20)   # scripts/lorra_circuit_breaker_llama3_8b.sh


def _proj_module(decoder_layer, name: str):
    """The Linear for a projection name on a decoder block, or None if the
    architecture lacks it (q/k/v/o under self_attn; gate/up/down under mlp)."""
    parent = decoder_layer.self_attn if name.endswith("o_proj") or name in (
        "q_proj", "k_proj", "v_proj") else decoder_layer.mlp
    return getattr(parent, name, None)


def resolve_layers(model, target_layers) -> tuple[list[int], list[int]]:
    """(target_layers, transform_layers) clamped to the model depth. Target
    layers carry the reroute loss; transform layers (0..max(target)) carry
    the LoRA adapters and the retain loss — faithful to the original's
    --target_layers / --transform_layers=-1 (== range(0, max_target+1))."""
    n = num_layers(model)
    tgt = sorted({min(int(l), n - 1) for l in target_layers})
    transform = list(range(0, max(tgt) + 1))
    return tgt, transform


class CircuitBreakerObject:
    """Trained RR adapter behind the stage-2 object protocol: a rank-r LoRA
    on the seven projections of layers 0..max(target), applied through
    forward hooks (frozen base). delta = scale · (α_lora/r) · (x·Aᵀ)·Bᵀ per
    projection. B is a tiny Gaussian at init (not PEFT's exact zero — see the
    init comment: exact-zero B makes the reroute gradient vanish); scale-0 is
    still a bit-exact no-op via the hook guard."""

    kind = "circuit_breaker"

    def __init__(self, model, transform_layers: list[int], rank: int,
                 lora_alpha: float, *, base_scale: float = 1.0,
                 target_layers: list[int] | None = None,
                 metadata: dict | None = None, seed: int = 0,
                 _params: dict | None = None):
        self.transform_layers = list(transform_layers)
        self.rank = int(rank)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / self.rank
        self.base_scale = float(base_scale)
        self.target_layers = list(target_layers or [])
        self.metadata = metadata or {}
        decoder = get_decoder_layers(model)
        self._modules: dict[tuple[int, str], nn.Module] = {}
        self.A: dict[tuple[int, str], nn.Parameter] = {}
        self.B: dict[tuple[int, str], nn.Parameter] = {}
        for l in self.transform_layers:
            for name in PROJECTIONS:
                mod = _proj_module(decoder[l], name)
                if mod is None:
                    continue
                key = (l, name)
                self._modules[key] = mod
                d_in = mod.weight.shape[1]
                d_out = mod.weight.shape[0]
                if _params is not None:
                    self.A[key] = nn.Parameter(_params["A"][key].clone())
                    self.B[key] = nn.Parameter(_params["B"][key].clone())
                else:
                    # Deterministic per-(layer, projection) seed — a fixed
                    # projection index, never Python's per-process-salted hash().
                    g = torch.Generator().manual_seed(
                        seed + l * len(PROJECTIONS) + PROJECTIONS.index(name))
                    # A: PEFT kaiming-uniform (a=sqrt(5) ⇒ bound 1/sqrt(fan_in)).
                    # B: a TINY Gaussian (σ=1e-3), NOT PEFT's exact zero — a
                    # deliberate deviation. With B=0 the delta is exactly 0, so
                    # h_lora == h_orig and the reroute loss ReLU(cos(h_lora,
                    # h_orig)) sits at cos=1, a flat maximum whose gradient wrt
                    # the LoRA params is EXACTLY 0 in exact arithmetic — the
                    # method cannot start learning (the original only escapes
                    # via bf16 rounding noise). A tiny nonzero B breaks the
                    # symmetry deterministically in float32/CPU too. scale-0
                    # stays a bit-exact no-op via the hook guard, so the
                    # object protocol is unaffected.
                    bound = 1.0 / math.sqrt(d_in)
                    a = torch.empty(self.rank, d_in).uniform_(-bound, bound, generator=g)
                    self.A[key] = nn.Parameter(a)
                    b = torch.empty(d_out, self.rank).normal_(0.0, 1e-3, generator=g)
                    self.B[key] = nn.Parameter(b)

    @property
    def trainable(self) -> list[nn.Parameter]:
        return [self.A[k] for k in self._modules] + [self.B[k] for k in self._modules]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.trainable)

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def _make_hook(self, key, scale: float):
        A, B = self.A[key], self.B[key]

        def hook(_mod, inputs, output):
            if scale == 0.0:
                return output
            x = inputs[0]
            delta = (x.to(A.dtype) @ A.t()) @ B.t()      # [..., d_out]
            return output + (scale * self.scaling * delta).to(output.dtype)

        return hook

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mag = self.base_scale * scale
        handles = [
            mod.register_forward_hook(self._make_hook(key, mag))
            for key, mod in self._modules.items()
        ]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def to_(self, device) -> None:
        for group in (self.A, self.B):
            for k in group:
                group[k].data = group[k].data.to(device)

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind,
             "transform_layers": list(self.transform_layers),
             "target_layers": list(self.target_layers),
             "rank": self.rank,
             "lora_alpha": self.lora_alpha,
             "base_scale": self.base_scale,
             "A": {f"{l}.{name}": self.A[(l, name)].detach().float().cpu()
                   for (l, name) in self._modules},
             "B": {f"{l}.{name}": self.B[(l, name)].detach().float().cpu()
                   for (l, name) in self._modules},
             "metadata": self.metadata},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "CircuitBreakerObject":
        def _unkey(d):
            out = {}
            for k, v in d.items():
                l, name = k.split(".", 1)
                out[(int(l), name)] = v
            return out

        params = {"A": _unkey(blob["A"]), "B": _unkey(blob["B"])}
        return cls(
            model, transform_layers=[int(l) for l in blob["transform_layers"]],
            rank=int(blob["rank"]), lora_alpha=float(blob["lora_alpha"]),
            base_scale=float(blob["base_scale"]),
            target_layers=[int(l) for l in blob.get("target_layers", [])],
            metadata=blob.get("metadata", {}), _params=params)


@contextmanager
def _capture(model, layers: list[int], detach: bool):
    """Capture decoder-block outputs {layer: hidden [B,S,d]}. detach=True for
    the frozen reference pass; detach=False keeps the graph for the adapted
    pass (capture.collect_pooled_activations always detaches, so it cannot be
    reused for the gradient forward)."""
    captured: dict[int, torch.Tensor] = {}
    decoder = get_decoder_layers(model)
    handles = []

    def make(idx):
        def hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[idx] = h.detach() if detach else h
        return hook

    for idx in layers:
        handles.append(decoder[idx].register_forward_hook(make(idx)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _encode(tok, texts: list[str], max_len: int, device):
    prev = tok.padding_side
    tok.padding_side = "right"
    try:
        enc = tok(texts, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=max_len)
    finally:
        tok.padding_side = prev
    return {k: v.to(device) for k, v in enc.items()}


def _retain_loss(model, obj, inputs, transform_layers,
                 denom: torch.Tensor | None = None) -> torch.Tensor:
    """mean_tokens ‖h_lora − h_orig‖₂ over transform layers, masked to real
    tokens (a proper masked mean — see divergence #1). ``denom`` overrides the
    normaliser: the micro-batched trainer passes the FULL batch's masked token
    count × n_layers, so per-chunk losses SUM to exactly the full-batch mean
    (the 156649 OOM fix — see the training loop)."""
    mask = inputs["attention_mask"].unsqueeze(0)  # [1,B,S] broadcast over layers
    with torch.no_grad():
        with _capture(model, transform_layers, detach=True) as orig:
            model(**inputs)
    h_orig = torch.stack([orig[l] for l in transform_layers]).float()
    with obj.apply(model, scale=1.0):
        with _capture(model, transform_layers, detach=False) as lora:
            model(**inputs)
    h_lora = torch.stack([lora[l] for l in transform_layers]).float()
    per_tok = (h_lora - h_orig).norm(dim=-1)          # [L,B,S]
    m = mask.to(per_tok.dtype)
    if denom is None:
        denom = (m.sum() * len(transform_layers)).clamp(min=1)
    return (per_tok * m).sum() / denom


def _reroute_loss(model, obj, inputs, target_layers,
                  denom: torch.Tensor | None = None) -> torch.Tensor:
    """mean_tokens ReLU(cos(h_lora, h_orig)) at the target layers, masked.
    ``denom`` overrides the normaliser (full-batch token count) so per-chunk
    losses sum to the full-batch mean — see _retain_loss."""
    mask = inputs["attention_mask"].unsqueeze(0)
    with torch.no_grad():
        with _capture(model, target_layers, detach=True) as orig:
            model(**inputs)
    h_orig = torch.stack([orig[l] for l in target_layers]).float()
    with obj.apply(model, scale=1.0):
        with _capture(model, target_layers, detach=False) as lora:
            model(**inputs)
    h_lora = torch.stack([lora[l] for l in target_layers]).float()
    cos = F.cosine_similarity(h_lora, h_orig, dim=-1)   # [L,B,S]
    m = mask.to(cos.dtype)
    if denom is None:
        denom = m.sum().clamp(min=1)
    return (F.relu(cos) * m).sum() / denom


def _micro_chunks(inputs: dict, size: int):
    """Split an encoded batch dict into micro-batches along dim 0 (views —
    the tensors are already on device)."""
    n = inputs["input_ids"].shape[0]
    for i in range(0, n, size):
        yield {k: v[i:i + size] for k, v in inputs.items()}


def train_circuit_breaker(
    model,
    tok,
    instruction,
    pairs,
    *,
    layer: int,
    log_fn=None,
    retain_texts: list[str] | None = None,
    reroute_texts: list[str] | None = None,
    allow_pairs_fallback: bool = False,
    target_layers=DEFAULT_TARGET_LAYERS,
    rank: int = 16,
    lora_alpha: float = 16.0,
    alpha: float = 10.0,
    lr: float = 1e-4,
    max_steps: int = 150,
    batch_size: int = 16,
    micro_batch_size: int = 4,
    max_len: int = 1024,
    seed: int = 0,
    epochs: int | None = None,   # train_kwargs.default compat (documented no-op)
) -> CircuitBreakerObject:
    """Registry trainer: rank-r LoRA over the seven projections of layers
    0..max(target), trained with the retain-L2 + reroute-ReLU(cos) losses
    under the α·progress / α·(1−progress) schedule.

    retain_texts (benign, already rendered prompt+completion) and
    reroute_texts (harmful prompt + HARMFUL continuation, rendered) are the
    PRODUCTION path. Absent them the trainer refuses unless
    allow_pairs_fallback=True — the fallback uses retain=render(prompt)+pos
    and reroute=render(prompt)+neg from the pairs and records
    fit_path="pairs_fallback_si_derived" (loud, never silent). epochs is a
    documented no-op (this trainer counts steps, not epochs).

    micro_batch_size (the 156649 OOM fix): each loss's batch is run through
    the model in micro-chunks with backward() per chunk, so at most ONE
    chunk's autograd graph is ever alive. The optimization is UNCHANGED —
    the sampled batch, the loss value, and the accumulated gradient are the
    full-batch ones exactly (per-chunk masked SUMS divided by the FULL
    batch's token count; fp summation order aside). micro_batch_size >=
    batch_size degenerates to the original single-pass behaviour."""
    del epochs
    device = next(model.parameters()).device
    if retain_texts is not None and reroute_texts is not None:
        fit_path = "explicit_kwargs"
    else:
        if not allow_pairs_fallback:
            raise ValueError(
                "circuit_breaker needs retain_texts (benign prompt+completion) "
                "and reroute_texts (harmful prompt + harmful CONTINUATION), "
                "already rendered. None supplied and allow_pairs_fallback=False "
                "— refusing to silently derive an SI-conditional fit from the "
                "pairs (a Class B object must be SI-agnostic). Pass the texts, "
                "or set allow_pairs_fallback=True for the recorded fallback.")
        fit_path = "pairs_fallback_si_derived"
        retain_texts = [build_chat_prompt(tok, p.prompt, system=None) + p.pos
                        for p in pairs]
        reroute_texts = [build_chat_prompt(tok, p.prompt, system=None) + p.neg
                         for p in pairs]
    if not retain_texts or not reroute_texts:
        raise ValueError("circuit_breaker got an empty retain or reroute set")

    tgt, transform = resolve_layers(model, target_layers)
    obj = CircuitBreakerObject(model, transform, rank, lora_alpha,
                               target_layers=tgt, seed=seed)
    obj.to_(device)
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(obj.trainable, lr=lr, weight_decay=0.0)

    rng = torch.Generator().manual_seed(seed)

    def _sample(texts, n):
        idx = torch.randint(0, len(texts), (n,), generator=rng).tolist()
        return _encode(tok, [texts[i] for i in idx], max_len, device)

    micro = max(1, min(int(micro_batch_size), batch_size))

    history: list[float] = []
    model.eval()  # frozen base; LoRA params train (dropout/BN would be base-side)
    for step in range(max_steps):
        progress = step / (2.0 * max_steps)          # original: step/(2·max_steps)
        retain_coeff = alpha * progress
        cb_coeff = alpha * (1.0 - progress)
        retain_inputs = _sample(retain_texts, batch_size)
        reroute_inputs = _sample(reroute_texts, batch_size)

        # OOM fix (job 156649: Llama-8B at B=16, S=1024 needs >135 GiB — the
        # adapted-forward graph saves an fp32 copy of every hooked projection
        # input across 21 layers, and the retain graph was held alive while
        # the reroute forward ran). Backward PER MICRO-CHUNK against the FULL
        # batch's denominator: gradients accumulate to exactly the full-batch
        # loss gradient (same rng draw, same samples, same effective batch —
        # the optimization is unchanged, fp summation order aside), while at
        # most one chunk's graph is alive at any moment.
        opt.zero_grad()
        r_val = c_val = 0.0
        if retain_coeff > 0:
            denom_r = (retain_inputs["attention_mask"].sum()
                       * len(transform)).clamp(min=1)
            for chunk in _micro_chunks(retain_inputs, micro):
                r = _retain_loss(model, obj, chunk, transform, denom=denom_r)
                (retain_coeff * r).backward()
                r_val += float(r.detach())
        if cb_coeff > 0:
            denom_c = reroute_inputs["attention_mask"].sum().clamp(min=1)
            for chunk in _micro_chunks(reroute_inputs, micro):
                c = _reroute_loss(model, obj, chunk, tgt, denom=denom_c)
                (cb_coeff * c).backward()
                c_val += float(c.detach())

        opt.step()
        loss_val = retain_coeff * r_val + cb_coeff * c_val
        history.append(loss_val)
        if log_fn:
            log_fn({"circuit_breaker/loss": loss_val,
                    "circuit_breaker/retain": r_val,
                    "circuit_breaker/reroute": c_val,
                    "circuit_breaker/retain_coeff": retain_coeff,
                    "circuit_breaker/cb_coeff": cb_coeff}, step)

    obj.metadata = {
        "paper": PAPER,
        "fit_path": fit_path,
        "target_layers": tgt,
        "transform_layers": transform,
        "rank": rank,
        "lora_alpha": lora_alpha,
        "alpha": alpha,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "micro_batch_size": micro,
        "max_len": max_len,
        "lr": lr,
        "n_retain": len(retain_texts),
        "n_reroute": len(reroute_texts),
        "final_loss": history[-1] if history else None,
    }
    return obj
