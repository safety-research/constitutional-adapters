"""LoFiT (arXiv 2406.01563, Yin, Ye & Durrett, NeurIPS 2024) as an
SI steering arm: two-phase localized fine-tuning on attention-head outputs.

FIDELITY SOURCES (verified 2026-08-17 against the official repo,
github.com/fc2869/lo-fit @ HEAD — models/modeling_llama.py, lofit_trainer.py,
utils/trainers.py, train_script_{truthfulqa,mquake,clutrr}.sh):

The method, exactly as their code runs it:
1. HEAD-SELECTION PHASE ("lofit_component=A"): per-head multiplicative
   scalers A_h — a HEAD_DIM VECTOR per head, not a scalar
   (modeling_llama.py L258: ``nn.Parameter(torch.zeros(self.head_dim))``
   per head) — applied to the attention output BEFORE o_proj:
       z_h  <-  (1 + A_h) ⊙ z_h  (+ v_h, zero in this phase)
   (modeling_llama.py L366-369: the per-head slice of the reshaped
   [B, S, n_heads*head_dim] attention output is multiplied by (A+1) and
   offset by v, THEN o_proj is applied). All heads' A train jointly, init
   N(0, 1e-3) (lofit_trainer.py L286), loss = task LM loss on the response
   tokens only (DataCollatorForCompletionOnlyLM) + l1_lambda * sum|A| over
   the trainable scalers (utils/trainers.py CustomSFTTrainer.compute_loss).
   After training, heads are RANKED by the L2 norm of each learned A_h
   (lofit_trainer.py L303: ``np.linalg.norm(attn_A[j])``, argsort
   descending) and the ranking is saved.
2. BIAS-TUNING PHASE ("lofit_component=v"): the model is loaded FRESH
   (A back to zeros -> the multiplication is identity), and per-head
   additive bias vectors v_h [head_dim] are trained ONLY for the top-K
   selected heads (all other parameters frozen), init N(0, 1e-3), same
   completion-only LM loss, NO L1 (their phase-v scripts pass
   --l1_lambda 0). The shipped artifact is therefore exactly
   "v_h added to the selected heads' outputs, BEFORE o_proj" — the
   selection scalers are discarded after ranking.

K CONVENTION (their scripts; the paper abstract says "a sparse set of
heads (3%)"): the phase-A ranking is saved generously (top 160 heads for
every model), but the K that defines the method is the BIAS-TUNED count:
llama2-7B -> 32 of 32x32=1024 heads (3.1%), llama2-13B -> 48 of 1600 (3%),
gemma-7b -> 16 of 448 (3.6%). It is a COUNT, chosen ~3% of total heads.
We default to their llama2-7B fraction exactly (32/1024, BIAS_HEAD_FRAC),
which on Llama-3.1-8B (32 layers x 32 heads) is again 32 heads; the full
[n_layers, n_heads] scaler-magnitude matrix is persisted in the artifact
(their saved superset ranking is subsumed by persisting ALL magnitudes).

Their hyperparameters (llama2-7B rows of the train scripts): phase A
lr 5e-3 (tqa/mquake; clutrr 5e-4), l1_lambda 5e-4..5e-3 (mquake, the
SFT task closest to ours: 1e-3 — our default); phase v lr 1e-2 (tqa fold-0
/ mquake / clutrr; 2e-2 elsewhere); both phases 5 epochs, batch 8, AdamW
via HF Trainer (weight_decay 0 — we pass weight_decay=0.0 explicitly since
torch's AdamW defaults to 0.01).

KEY DELIBERATE DEVIATION FROM THE RECENT ARMS — NO LAYER-15 PIN:
every other gradient arm in this campaign intervenes at the pinned layer
15; LoFiT's head selection ranges over ALL layers x ALL heads, and that
localization is the method's identity (the paper: "the localization step
is important"). It is exactly what makes this arm a complement to ReFT:
loreft edits residual positions at a pinned layer; lofit edits sparse
attention-head outputs wherever selection puts them. The registry's
``layer`` kwarg is accepted for signature uniformity and DELIBERATELY
ignored (recorded as ``pinned_layer_ignored`` in the metadata). The
selected (layer, head) map is itself a scientific result — where does
SI-following live? — and is persisted for plotting.

SI STEERING ADAPTATIONS (deliberate, mirroring the sibling arms):
- supervision: the arm's standard A1 pairs — the SI-prompted teacher's
  completions as targets for the UNPROMPTED student (the same
  DistillExample student side loreft trains on; LoFiT's own loss IS the
  task LM loss on targets, so this is their objective on our data);
- budget: epochs/batch_size/max_len come from the shared
  train_kwargs.default (6/2/768) so the comparison with lora_r1/loreft is
  like-for-like (theirs: 5 epochs, batch 8 — documented, not replicated);
- GQA: their models are MHA (llama2); on GQA models (Llama-3.1-8B: 32 Q
  heads over 8 KV heads) the per-head structure of the pre-o_proj
  attention output is the 32 QUERY heads, which is what we slice;
- base scale: the paper applies the biases at strength 1, so base_scale
  is FIXED at 1.0 (no calibration ladder); the eval factor grid sweeps
  around it and the canary's coherence guards (distinct-3 / 3x gen-ppl)
  check it, exactly like every other arm's grid.

Position semantics: the biases apply at EVERY position of every forward
(their modeling code bakes the addition into the module — prompt,
completion, and decode steps alike, BOS included); scale 0 detaches the
hooks entirely and is bit-exact unsteered.

PHASE-V DIVERGENCE POST-MORTEM (2026-08-24, si_constitution wave objects):
the all-positions fit on si_constitution shipped a 58.4x-floor object
(probe KL 4.28 vs no-op floor 0.0733) while phase-A ``select_loss`` sat
flat at ~0.49 for all 30,990 of its steps — the two phases are SEPARATE
optimizations, so phase-A metrics cannot see phase-v health. The per-step
``lofit/bias_ce`` log shows the actual failure: phase v ran healthy
(~0.35) through epochs 0-1, then fell off a cliff FIVE steps into epoch 2
(bias steps 10335→10336: 0.24 → 2.3 → 12.1) and sat in a ~7.0-CE basin
for the remaining 20,655 steps without recovering. Same code, same
budget, safety_refusal (all-positions AND response-fit) and the
si_constitution response-fit stayed at ~0.27-0.35 throughout — the cliff
is pool-dependent (an Adam instability at fixed lr 1e-2, unclipped), not
universal. Three guards fixed it, all defaults:

1. ``lr_decay`` / ``max_grad_norm``: their training ran under HF Trainer,
   whose TrainingArguments DEFAULT to linear LR decay (to 0) and
   grad-norm clipping at 1.0; this port replicated the lr/epochs/wd but
   ran FIXED-LR and UNCLIPPED — a fidelity gap, now closed (both phases).
2. Per-epoch held-out early stopping on the SAME soft-KL-to-teacher probe
   the driver's floor guard rules on (lockstep.arm_probe_losses), with
   best-epoch restore — the lockstep group's stop, wired for this solo
   trainer (phase v only: phase A ships a RANKING, not weights, and its
   loss was stable on every wave run). Even if a cliff still fires, the
   shipped object is the best-probe epoch, never the post-cliff one; a
   fit whose BEST epoch is degenerate still fails the floor guard loudly
   (EarlyStopper.restore_best never restores the init).
3. When no held-out set is passed, a small probe slice is self-carved
   from the pool (``heldout_frac``, capped at HELDOUT_MAX) — the wave
   driver trains solo arms through the registry and cannot pass one.
   Pools too small to carve (< HELDOUT_MIN held-out rows — every tiny
   test fixture) skip the carve and the stop entirely, keeping the
   historical fixed-budget path.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..hooks import PositionMaskedFit, SteeringHandle
from ..models import get_decoder_layers
from . import ddp as _ddp
from .distill import (DistillExample, EarlyStopper, EarlyStopSpec,
                      _encode_aligned, _gather_completion_logits,
                      loss_gather_masks, steer_prompt_boundaries)
from .traindata import Batch, collate

PAPER = "arXiv:2406.01563 (LoFiT, Yin/Ye/Durrett, NeurIPS 2024)"

# Their init for both phases: nn.init.normal_(module, mean=0, std=1e-3)
# (lofit_trainer.py L286/L292).
INIT_STD = 1e-3

# Their llama2-7B bias-tuning K: 32 of 1024 heads. Expressed as the exact
# fraction so other geometries inherit the same convention (~3%, the paper's
# "sparse set of heads (3%)"). On Llama-3.1-8B (32x32) this is again 32.
BIAS_HEAD_FRAC = 32 / 1024

# ---- trainer-side held-out probe guard (module docstring, post-mortem) -----
# Shared by train_lofit and reft.train_loreft (which imports these — they
# belong next to EarlyStopper in distill.py, but this fix's blast radius is
# lofit.py/reft.py only; move them on the next shared-edit window).
HELDOUT_FRAC = 0.05   # fraction of the pool self-carved for the probe
HELDOUT_MAX = 16      # the probe needs stability, not a big split
HELDOUT_MIN = 4       # below this the probe is noise: run the fixed budget
PROBE_BATCH = 4       # arm_probe_losses chunk width (~the driver's own)


def carve_heldout(
    examples: list[DistillExample], frac: float, seed: int,
    max_n: int = HELDOUT_MAX, min_n: int = HELDOUT_MIN,
) -> tuple[list[DistillExample], list[DistillExample]]:
    """Deterministically carve a small held-out probe slice from ``examples``
    for per-epoch early stopping. Returns ``(train, heldout)``. When the pool
    is too small for a meaningful probe (< ``min_n`` carved rows — every tiny
    test fixture), returns ``(examples, [])`` with the input list untouched
    and NO RNG drawn, so the historical fixed-budget path is bit-identical.
    The pick uses a fresh ``random.Random(seed)``, independent of the
    trainer's epoch-shuffle stream."""
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"heldout_frac must be in [0, 1), got {frac}")
    k = min(int(frac * len(examples)), max_n)
    if k < min_n:
        return examples, []
    pick = set(random.Random(seed).sample(range(len(examples)), k))
    train = [ex for i, ex in enumerate(examples) if i not in pick]
    held = [examples[i] for i in sorted(pick)]
    return train, held


def heldout_probe(model, tok, obj, heldout: list[DistillExample], *,
                  max_len: int,
                  batch_size: int = PROBE_BATCH) -> tuple[float, float]:
    """Mean ``(active, no-op floor)`` soft-KL-to-teacher probe over EVERY
    held-out row, via the SAME public probe the driver's floor guard rules
    on (``lockstep.arm_probe_losses``) — so the early-stop metric IS the
    guard's metric, measured on held-out rows. One call, ``n_probe=None``:
    the probe averages per-example values itself, so ``batch_size`` is
    memory width only (the old outer chunk loop averaged chunk means
    equally, weighting a ragged last chunk's examples more — the same
    chunk-dependence the 2026-08-24 estimator fix removed). No grad, no
    RNG draws, no param writes. AUTHORITATIVE AND UNCACHED like the probe
    it wraps: this number selects the restored epoch, so the teacher
    prefix cache — a gradient-training-only optimisation — deliberately
    has no path in."""
    from .lockstep import arm_probe_losses  # deferred; lofit needs no more

    act, floor = arm_probe_losses(model, tok, obj, heldout, max_len=max_len,
                                  batch_size=batch_size, n_probe=None)
    return act, floor


def _n_attention_heads(model) -> int:
    cfg = model.config
    if hasattr(cfg, "num_attention_heads"):
        return cfg.num_attention_heads
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, "num_attention_heads"):
        return text_cfg.num_attention_heads
    raise ValueError(
        f"no num_attention_heads on {type(cfg).__name__} — cannot slice "
        "the o_proj input into heads"
    )


def head_geometry(model) -> tuple[int, int, int]:
    """(n_layers, n_heads, head_dim), with head_dim derived from the REAL
    o_proj input width (num_heads * head_dim != hidden_size on some models
    — gemma-2-9b's o_proj is 4096->3584; sizing from hidden_size silently
    mis-slices, the same trap test_lora_sizes_from_oproj_not_hidden_size
    documents for the LoRA arms). Loud on non-divisibility."""
    layers = get_decoder_layers(model)
    n_heads = _n_attention_heads(model)
    d_in = layers[0].self_attn.o_proj.weight.shape[1]
    if d_in % n_heads != 0:
        raise ValueError(
            f"o_proj input width {d_in} is not divisible by "
            f"num_attention_heads {n_heads} — refusing to guess the head "
            "slicing"
        )
    return len(layers), n_heads, d_in // n_heads


def select_heads(magnitudes: torch.Tensor, k: int) -> list[tuple[int, int]]:
    """Top-k (layer, head) tuples by scaler magnitude, DESCENDING — their
    ranking convention (lofit_trainer.py L308:
    ``np.argsort(ahead.flatten())[::-1][:k]`` with flat index
    layer*num_heads + head)."""
    if magnitudes.dim() != 2:
        raise ValueError(f"magnitudes must be [n_layers, n_heads], "
                         f"got shape {tuple(magnitudes.shape)}")
    n_layers_, n_heads = magnitudes.shape
    total = n_layers_ * n_heads
    if not 0 < k <= total:
        raise ValueError(f"k must be in 1..{total}, got {k}")
    idx = torch.argsort(magnitudes.flatten(), descending=True)[:k]
    return [(int(i) // n_heads, int(i) % n_heads) for i in idx]


class LoFiTObject(PositionMaskedFit):
    """Per-head additive biases v_h [head_dim] on the selected heads'
    attention outputs, applied BEFORE o_proj (the shipped LoFiT artifact —
    the selection scalers are ranking-only and discarded, matching their
    phase-v model where attn_A is back at zeros).

    apply(model, scale): x[..., h*hd:(h+1)*hd] += scale * base_scale * v_h
    at every position of every forward. scale == 0 registers nothing and
    is bit-exact unsteered. base_scale is 1.0 by construction (the paper
    applies at strength 1); it is stored so the artifact convention
    matches the calibrated arms (antidote/did/adasteer/alphasteer).

    Position-masked fits (PositionMaskedFit): the bias is added to the
    pre-o_proj attention output — POST-attention-mix, so position t's
    addition reaches only position t's o_proj output, and o_proj is a
    position-wise linear map (the edit is exactly a per-position residual
    addition of W_o·scatter(v)). Restricting it to the handle's mask is
    therefore EXACT, with masked positions bit-identical to the unsteered
    model. apply() yields the live mask handle (None on the default
    all-positions fit).
    """

    kind = "lofit"

    def __init__(self, model, heads: list[tuple[int, int]],
                 base_scale: float = 1.0, init_std: float = INIT_STD,
                 seed: int = 0, metadata: dict | None = None):
        self.n_layers, self.n_heads, self.head_dim = head_geometry(model)
        seen: set[tuple[int, int]] = set()
        self.heads: list[tuple[int, int]] = []
        for l, h in heads:
            l, h = int(l), int(h)
            if not (0 <= l < self.n_layers and 0 <= h < self.n_heads):
                raise ValueError(
                    f"head ({l}, {h}) out of range for {self.n_layers} "
                    f"layers x {self.n_heads} heads"
                )
            if (l, h) in seen:
                raise ValueError(f"duplicate head ({l}, {h})")
            seen.add((l, h))
            self.heads.append((l, h))
        g = torch.Generator().manual_seed(seed)
        # Their phase-v init: N(0, 1e-3) on the tuned heads — deliberately
        # NOT the harness's zero-init convention (documented deviation like
        # reft_r1's 0.01 fill); scale=0 through apply() stays bit-exact.
        self.v = nn.Parameter(
            init_std * torch.randn(len(self.heads), self.head_dim,
                                   generator=g)
        )
        self.base_scale = float(base_scale)
        self.metadata = dict(metadata or {})
        # The full phase-A scaler-magnitude map [n_layers, n_heads] — the
        # "where does SI-following live?" result; persisted with the
        # artifact for plotting (train_lofit fills it).
        self.scaler_magnitudes: torch.Tensor | None = None
        self._decoder = get_decoder_layers(model)
        # Precompute, per involved layer, the parameter rows and their flat
        # positions in the o_proj input.
        rows_by_layer: dict[int, list[int]] = {}
        for row, (l, h) in enumerate(self.heads):
            rows_by_layer.setdefault(l, []).append(row)
        self._rows_by_layer = {
            l: torch.tensor(rows, dtype=torch.long)
            for l, rows in rows_by_layer.items()
        }
        self._pos_by_layer = {
            l: torch.cat([
                torch.arange(self.heads[r][1] * self.head_dim,
                             (self.heads[r][1] + 1) * self.head_dim)
                for r in rows
            ])
            for l, rows in rows_by_layer.items()
        }

    @property
    def trainable(self) -> list[nn.Parameter]:
        return [self.v]

    def n_params(self) -> int:
        return self.v.numel()

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        coef = float(scale) * self.base_scale
        if coef == 0.0 or not self.heads:
            yield  # exact no-op (the harness scale-0 invariant)
            return
        mask_handle = self._fit_mask_handle(model)

        def make_hook(rows: torch.Tensor, pos: torch.Tensor):
            def pre_hook(_mod, args):
                x = args[0]
                src = (coef * self.v[rows].to(torch.float32)).reshape(-1)
                delta = torch.zeros(x.shape[-1], dtype=torch.float32,
                                    device=x.device)
                delta = delta.index_copy(0, pos.to(x.device),
                                         src.to(x.device))
                steered = self._fit_masked(mask_handle, x + delta.to(x.dtype),
                                           x)
                return (steered,) + args[1:]

            return pre_hook

        handles = [
            self._decoder[l].self_attn.o_proj.register_forward_pre_hook(
                make_hook(self._rows_by_layer[l], self._pos_by_layer[l]))
            for l in self._rows_by_layer
        ]
        try:
            yield mask_handle
        finally:
            for h in handles:
                h.remove()
            if mask_handle is not None:
                mask_handle.remove()

    def to_(self, device) -> None:
        self.v.data = self.v.data.to(device)

    def save(self, path: Path) -> None:
        torch.save(
            {
                "kind": self.kind,
                "heads": torch.tensor(self.heads, dtype=torch.long),
                "v": self.v.detach().float().cpu(),
                "base_scale": self.base_scale,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "head_dim": self.head_dim,
                "scaler_magnitudes": (
                    None if self.scaler_magnitudes is None
                    else self.scaler_magnitudes.detach().float().cpu()
                ),
                # The FIT's mask leg is part of the artifact's identity (the
                # DiffMeanObject/LoReFT precedent, 2026-08-21 — supersedes
                # the earlier fit-time-only stance: before these keys a
                # response-fit lofit re-hydrated ALL-POSITION and would
                # silently run the all-token method under a decode-only
                # label). The trainer's metadata record of the same leg
                # stays as documentation; THESE keys are what load restores.
                "positions": self.fit_positions,
                "response_header_len": self.fit_response_header_len,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "LoFiTObject":
        n_layers_, n_heads, head_dim = head_geometry(model)
        got = (int(blob["n_layers"]), int(blob["n_heads"]),
               int(blob["head_dim"]))
        if got != (n_layers_, n_heads, head_dim):
            raise ValueError(
                f"lofit artifact geometry {got} (layers, heads, head_dim) "
                f"does not match this model's ({n_layers_}, {n_heads}, "
                f"{head_dim}) — the head map is model-specific; refusing "
                "to apply it to a different architecture"
            )
        heads = [(int(l), int(h)) for l, h in blob["heads"].tolist()]
        obj = cls(model, heads, base_scale=float(blob["base_scale"]),
                  metadata=blob.get("metadata") or {})
        obj.v.data = blob["v"].clone()
        mags = blob.get("scaler_magnitudes")
        obj.scaler_magnitudes = None if mags is None else mags.clone()
        # Backward-compatible fit-leg restore (the DiffMeanObject pattern):
        # no positions key -> the historical all-token fit, byte-identical.
        if blob.get("positions", "all") != "all":
            obj.set_fit_positions(blob["positions"],
                                  blob.get("response_header_len"))
        return obj


@contextmanager
def scaler_context(model, A: torch.Tensor,
                   mask_handle: SteeringHandle | None = None):
    """Phase-A forward path: at EVERY layer, the pre-o_proj attention
    output is multiplied per-head by (1 + A_l) — implemented as
    x + x ⊙ A_l on the flattened [n_heads*head_dim] width, computed in
    float32 like the sibling hook arms. A: [n_layers, n_heads, head_dim]
    (phase A trains all heads jointly; the phase's v is zero and elided).

    ``mask_handle`` (a vectorless hooks.SteeringHandle, caller-owned)
    restricts the scalers to its mask — the decode-only fit leg, so head
    IMPORTANCE is ranked under the same position support the phase-v
    biases will train on. Masked positions stay bit-identical. The
    multiplicative edit is per-position for the same reason the bias is
    (post-attention-mix, position-wise o_proj)."""
    decoder = get_decoder_layers(model)
    if A.shape[0] != len(decoder):
        raise ValueError(
            f"A has {A.shape[0]} layers but the model has {len(decoder)}"
        )

    def make_hook(l: int):
        def pre_hook(_mod, args):
            x = args[0]
            mult = A[l].reshape(-1).to(torch.float32)
            delta = x.to(torch.float32) * mult
            steered = x + delta.to(x.dtype)
            steered = PositionMaskedFit._fit_masked(mask_handle, steered, x)
            return (steered,) + args[1:]

        return pre_hook

    handles = [
        decoder[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))
        for l in range(A.shape[0])
    ]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _ce_on_completions(logits: torch.Tensor, batch: Batch,
                       gather: torch.Tensor | None = None) -> torch.Tensor:
    """Completion-only LM cross-entropy — their
    DataCollatorForCompletionOnlyLM objective on our aligned batches.
    ``gather`` ([B, S-1] bool, from loss_gather_masks) restricts the loss
    to the completion positions a fit mask covers; None keeps the
    historical full-completion gather bit-for-bit."""
    s_logits = _gather_completion_logits(logits, batch, gather=gather)
    labels = batch.labels[:, 1:]
    targets = labels[gather] if gather is not None else labels[labels != -100]
    return F.cross_entropy(s_logits.float(), targets)


def _check_finite(loss: torch.Tensor, phase: str) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"lofit {phase} loss is not finite ({float(loss)}) — refusing "
            "to continue training on a diverged objective"
        )


def _log_head_map(log_fn, magnitudes: torch.Tensor,
                  heads: list[tuple[int, int]], step: int) -> None:
    """The selected-head map as a W&B image (figures rule: if a diagnostic
    is worth computing, plot and log it). Scalars-only log_fns simply get
    an Image value; W&B renders it in the Media panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(magnitudes.numpy(), aspect="auto", cmap="viridis")
    ls, hs = zip(*heads)
    ax.scatter(hs, ls, s=18, facecolors="none", edgecolors="red",
               label=f"selected ({len(heads)})")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("LoFiT phase-A scaler magnitudes ||A_h||")
    fig.colorbar(im, ax=ax, label="||A_h||")
    ax.legend(loc="upper right", fontsize=7)
    log_fn({"lofit/head_map": wandb.Image(fig)}, step)
    plt.close(fig)


def train_lofit(
    model,
    tok,
    examples: list[DistillExample],
    *,
    layer: int | None = None,
    instruction_id: str = "",
    lr_select: float = 5e-3,
    lr_bias: float = 1e-2,
    l1_lambda: float = 1e-3,
    n_bias_heads: int | None = None,
    epochs: int = 6,
    batch_size: int = 2,
    max_len: int = 768,
    init_std: float = INIT_STD,
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
) -> LoFiTObject:
    """The two-phase LoFiT trainer (module docstring for fidelity notes).

    Divergence guards (module docstring's post-mortem — all defaults):
    ``lr_decay`` (linear to 0 per phase) and ``max_grad_norm`` (clip; None
    disables) restore the HF-Trainer defaults the reference runs trained
    under. ``heldout`` + ``early_stop`` wire the lockstep group's per-epoch
    held-out stop into phase v: after every bias epoch the object is probed
    with the floor guard's own soft-KL-to-teacher metric on the held-out
    rows, and the SHIPPED weights are the best epoch's (``EarlyStopper``,
    patience per the spec; the cap-reached case restores the best epoch
    too). When ``heldout`` is None a probe slice is self-carved from
    ``examples`` (``heldout_frac`` of the pool, HELDOUT_MAX cap); pools too
    small to carve (< HELDOUT_MIN rows) skip the carve and the stop
    entirely — the historical fixed-budget path, bit-for-bit. Phase A keeps
    the fixed budget deliberately: it ships a RANKING, not weights, and its
    loss was stable on every wave run. ``diagnostics`` (optional dict) is
    filled with the probe curve and stopping record.

    ``layer`` is accepted for the registry's uniform trainer signature and
    DELIBERATELY IGNORED — the head selection ranges over ALL layers (the
    method's identity and this arm's key deviation from the layer-15-pinned
    siblings); it is recorded as ``pinned_layer_ignored`` in the metadata
    so the artifact is self-documenting.

    ``positions="response"`` selects the decode-only fit leg: BOTH phases
    apply their edit only from each row's boundary on (header-steered when
    ``response_header_len`` is set; H is derived ONCE by the caller, never
    here) — so the phase-A head ranking is measured under the same
    position support the phase-v biases train on — and each phase's CE
    gathers exactly the completion positions the mask covers
    (``loss_gather_masks``). The default is byte-identical to the
    historical all-positions fit.
    """
    if not examples:
        raise ValueError("lofit got no training examples")
    if positions not in ("all", "response"):
        raise ValueError(
            f"lofit fit positions must be 'all' or 'response', got "
            f"{positions!r}")
    device = next(model.parameters()).device
    n_layers_, n_heads, head_dim = head_geometry(model)
    total_heads = n_layers_ * n_heads
    if n_bias_heads is None:
        n_bias_heads = max(1, round(BIAS_HEAD_FRAC * total_heads))
    if not 0 < n_bias_heads <= total_heads:
        raise ValueError(
            f"n_bias_heads must be in 1..{total_heads}, got {n_bias_heads}"
        )
    torch.manual_seed(seed)
    rng = random.Random(seed)
    for p in model.parameters():
        p.requires_grad_(False)

    # Held-out probe slice for the phase-v stop: an explicit ``heldout``
    # wins; otherwise self-carve (no-op on pools too small to probe).
    if heldout is None:
        examples, heldout = carve_heldout(examples, heldout_frac, seed)

    # Student side only (no-SI prompt + teacher completion) — the same data
    # diet as train_loreft, and LoFiT's own loss is the task LM loss.
    encoded = [_encode_aligned(tok, ex, max_len)[1] for ex in examples]
    batches: list[Batch] = [
        collate(encoded[i: i + batch_size], tok.pad_token_id)
        for i in range(0, len(encoded), batch_size)
    ]

    # ---- phase A: head selection ------------------------------------------
    A = nn.Parameter(torch.empty(n_layers_, n_heads, head_dim,
                                 device=device))
    with torch.no_grad():
        A.normal_(mean=0.0, std=init_std)  # their phase-A init
    # HF Trainer's AdamW runs weight_decay 0 (their TrainingArguments leave
    # the default); torch's AdamW defaults to 0.01, which would add an L2
    # confound to the L1 selection pressure — pass 0 explicitly.
    opt = torch.optim.AdamW([A], lr=lr_select, weight_decay=0.0)
    # HF Trainer's OTHER two relevant defaults, previously dropped by this
    # port (the post-mortem's fidelity gap): linear LR decay to 0 over the
    # phase's step budget, and grad-norm clipping at max_grad_norm.
    sched = None
    if lr_decay:
        total_a = max(epochs * len(batches), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: max(0.0, 1.0 - s / total_a))
    # The decode-only leg's ONE mask source for phase A (phase v gets its
    # own from the object's apply()): a vectorless SteeringHandle, per-row
    # boundaries set per batch, CE tied to the same mask.
    select_mask_handle = None
    if positions != "all":
        select_mask_handle = SteeringHandle(
            model, {}, positions=positions,
            response_header_len=response_header_len)
    # Row-sharded fit context (extraction.ddp): each rank forwards only its
    # row slice of every batch — same batches, order, effective batch and
    # step count as single-process (the exact-semantics contract); byte-
    # identical single-process path when no context is set. The L1 term is
    # data-INDEPENDENT and identical on every rank, so scaling the whole
    # local loss by k/K still sums to exactly one copy of it.
    par = _ddp.active()
    sharded = par is not None and par.enabled

    def _ce_count(b, s_gather) -> int:
        if s_gather is not None:
            return int(s_gather.sum())
        return int((b.labels[:, 1:] != -100).sum())

    select_history: list[float] = []
    step = 0
    try:
        for epoch in range(epochs):
            order = list(range(len(batches)))
            rng.shuffle(order)
            for bi in order:
                b_full = batches[bi]
                rows = _ddp.rows_for_rank(b_full.input_ids.shape[0], par)
                loss = None
                k_local = 0
                if rows:
                    b = (_ddp.slice_batch(b_full, rows) if sharded
                         else b_full).to(device)
                    s_gather = None
                    if select_mask_handle is not None:
                        select_mask_handle.set_prompt_lens(
                            steer_prompt_boundaries(b, response_header_len))
                    with scaler_context(model, A,
                                        mask_handle=select_mask_handle):
                        logits = model(input_ids=b.input_ids,
                                       attention_mask=b.attention_mask).logits
                        if select_mask_handle is not None:
                            s_gather, _tg, _keep = loss_gather_masks(
                                select_mask_handle, b)
                    ce = _ce_on_completions(logits, b, gather=s_gather)
                    l1 = A.abs().sum()
                    loss = ce + l1_lambda * l1
                    _check_finite(loss, "phase-A (selection)")
                    k_local = _ce_count(b, s_gather)
                opt.zero_grad()
                if sharded:
                    (k_glob,) = _ddp.allreduce_counts([k_local], par)
                    if loss is not None:
                        scaled = _ddp.scale_loss_parts(
                            [(loss, k_local, k_glob)], device)
                        scaled.backward()
                        share = float(scaled.detach())
                    else:
                        share = 0.0
                    (loss_val,) = _ddp.allreduce_grads_and_stats(
                        [A], [share], par)
                else:
                    loss.backward()
                    loss_val = float(loss.detach())
                # Clip AFTER the all-reduce: the global gradient every rank
                # now holds is the tensor the single-process clip sees
                # (clipping local partials before the sum would not be).
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_([A], max_grad_norm)
                opt.step()
                if sched is not None:
                    sched.step()
                step += 1
                select_history.append(loss_val)
                if log_fn and loss is not None:
                    log_fn({"lofit/select_loss": loss_val,
                            "lofit/select_ce": float(ce.detach()),
                            "lofit/select_l1": float(l1.detach()),
                            "epoch": epoch}, step)
    finally:
        if select_mask_handle is not None:
            select_mask_handle.remove()
    if sharded:
        _ddp.assert_replicas_equal("lofit/phase-A", [A], par)

    magnitudes = A.detach().float().norm(dim=-1).cpu()  # [n_layers, n_heads]
    if float(magnitudes.max()) == 0.0:
        raise RuntimeError(
            "lofit phase-A scalers are all exactly zero after training — "
            "the selection ranking would be arbitrary; refusing to pick "
            "heads from no signal"
        )
    heads = select_heads(magnitudes, n_bias_heads)
    del A

    # ---- phase v: bias tuning on the selected heads ------------------------
    # Fresh biases, their N(0, 1e-3) init; the scalers are DISCARDED (their
    # phase-v model reloads with attn_A at zeros), surviving only as the
    # persisted magnitude map.
    obj = LoFiTObject(model, heads, base_scale=1.0, init_std=init_std,
                      seed=seed)
    if positions != "all":
        obj.set_fit_positions(positions, response_header_len)
    obj.to_(device)
    opt = torch.optim.AdamW(obj.trainable, lr=lr_bias, weight_decay=0.0)
    sched = None
    if lr_decay:
        total_v = max(epochs * len(batches), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: max(0.0, 1.0 - s / total_v))
    bias_history: list[float] = []
    bias_step = 0
    # The phase-v stop (module docstring's post-mortem): probe with the
    # guard's own metric after every epoch, ship the best epoch's biases.
    stopper = EarlyStopper(early_stop or EarlyStopSpec()) if heldout else None
    heldout_curve: list[float] = []
    probe_floor: float | None = None
    stopped_epoch: int | None = None
    bias_epochs_run = 0

    def _probe(epoch: int) -> float:
        nonlocal probe_floor
        act, floor = heldout_probe(model, tok, obj, heldout, max_len=max_len)
        probe_floor = floor
        heldout_curve.append(act)
        if log_fn:
            log_fn({"lofit/heldout_probe": act,
                    "lofit/heldout_probe_floor": floor,
                    "epoch": epoch}, step + bias_step)
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
            ce = None
            k_local = 0
            if rows:
                b = (_ddp.slice_batch(b_full, rows) if sharded
                     else b_full).to(device)
                s_gather = None
                with obj.apply(model, scale=1.0) as mask_handle:
                    if mask_handle is not None:
                        mask_handle.set_prompt_lens(
                            steer_prompt_boundaries(b, response_header_len))
                    logits = model(input_ids=b.input_ids,
                                   attention_mask=b.attention_mask).logits
                    if mask_handle is not None:
                        s_gather, _tg, _keep = loss_gather_masks(
                            mask_handle, b)
                ce = _ce_on_completions(logits, b, gather=s_gather)
                _check_finite(ce, "phase-v (bias tuning)")
                k_local = _ce_count(b, s_gather)
            opt.zero_grad()
            if sharded:
                (k_glob,) = _ddp.allreduce_counts([k_local], par)
                if ce is not None:
                    scaled = _ddp.scale_loss_parts(
                        [(ce, k_local, k_glob)], device)
                    scaled.backward()
                    share = float(scaled.detach())
                else:
                    share = 0.0
                (ce_val,) = _ddp.allreduce_grads_and_stats(
                    obj.trainable, [share], par)
            else:
                ce.backward()
                ce_val = float(ce.detach())
            # Clip AFTER the all-reduce (see phase-A note above).
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(obj.trainable, max_grad_norm)
            opt.step()
            if sched is not None:
                sched.step()
            bias_step += 1
            bias_history.append(ce_val)
            if log_fn:
                log_fn({"lofit/bias_ce": ce_val,
                        "epoch": epoch}, step + bias_step)
        bias_epochs_run = epoch + 1
        if stopper is not None:
            act = _probe(epoch)
            if stopper.update(act, obj.trainable):
                stopper.restore_best(obj.trainable)
                stopped_epoch = epoch + 1
                if log_fn:
                    log_fn({"lofit/stopped_epoch": stopped_epoch,
                            "lofit/best_epoch": stopper.best_epoch},
                           step + bias_step)
                break
    if stopper is not None and stopped_epoch is None:
        # Budget cap reached with a better epoch inside the patience
        # window: ship the best epoch, not the last (lockstep's rule).
        stopper.restore_best(obj.trainable)
    if sharded:
        # After the restore, so the assertion rules on the params that SHIP
        # (a rank-divergent probe decision would surface here, loudly).
        _ddp.assert_replicas_equal("lofit/phase-v", obj.trainable, par)

    obj.scaler_magnitudes = magnitudes
    obj.metadata.update({
        "paper": PAPER,
        "instruction_id": instruction_id,
        "model_name": getattr(model.config, "_name_or_path", ""),
        # the head map, duplicated as plain lists so the sidecar-less blob
        # is self-describing without tensor decoding
        "selected_heads": [[l, h] for l, h in heads],
        "selected_magnitudes": [float(magnitudes[l, h]) for l, h in heads],
        "layers_spanned": sorted({l for l, _ in heads}),
        "n_bias_heads": n_bias_heads,
        "total_heads": total_heads,
        "bias_head_frac": n_bias_heads / total_heads,
        # THE deliberate deviation: no layer pin — selection ranged
        # over all layers; the registry's pinned layer is recorded, unused.
        "pinned_layer_ignored": layer,
        # The fit's mask leg, as documentation (since 2026-08-21 the same
        # leg is ALSO persisted at the blob's top level and restored by
        # load_blob — a response-fit artifact re-hydrates masked; serve
        # phase stays the eval cell's explicit choice via
        # jailbreak._set_response_positions).
        "fit_positions": positions,
        "fit_response_header_len": response_header_len,
        "lr_select": lr_select,
        "lr_bias": lr_bias,
        "l1_lambda": l1_lambda,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": max_len,
        "init_std": init_std,
        "seed": seed,
        "n_examples": len(examples),
        "n_heldout": len(heldout),
        "lr_decay": lr_decay,
        "max_grad_norm": max_grad_norm,
        "base_scale": 1.0,
        "base_scale_rule": ("paper applies the biases at strength 1; the "
                            "eval factor grid sweeps around it"),
        "n_params_bias": obj.n_params(),
        "n_params_selection_phase": n_layers_ * n_heads * head_dim,
        "final_select_loss": select_history[-1] if select_history else None,
        # The LAST step's loss, which after a best-epoch restore no longer
        # describes the shipped biases — the early_stop record does.
        "final_bias_ce": bias_history[-1] if bias_history else None,
    })
    es_record = {
        "enabled": stopper is not None,
        "metric": "soft-KL-to-teacher probe (lockstep.arm_probe_losses) "
                  "on held-out rows — the floor guard's own metric",
        "n_heldout": len(heldout),
        # curve[0] is the pre-training baseline (epoch -1), then one point
        # per completed bias epoch.
        "heldout_curve": heldout_curve,
        "noop_floor_probe": probe_floor,
        "best_epoch": stopper.best_epoch if stopper else None,
        "best_heldout_probe": (
            stopper.best if stopper and math.isfinite(stopper.best)
            else None),
        "stopped_epoch": stopped_epoch,
        "bias_epochs_run": bias_epochs_run,
    }
    obj.metadata["early_stop"] = es_record
    if diagnostics is not None:
        diagnostics.update(es_record)
    if log_fn:
        _log_head_map(log_fn, magnitudes, heads, step + bias_step + 1)
    return obj
