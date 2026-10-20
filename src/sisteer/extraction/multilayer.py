"""All-layer joint variants of the vector and rank-1-LoRA objects.

The layer-placement axis of stage 2: instead of one intervention at the
mid layer, ONE parameter set PER decoder layer, trained JOINTLY under the
same forward-KL objective (train_object_kl treats apply() as opaque, so
the summed-over-layers loss is implicit in the joint forward), and
evaluated with a SINGLE shared scale multiplying every layer.

- MultiLayerVectorObject: one d-vector per layer, zero-init (starts at
  the unsteered model), applied through the existing hooks.steering
  machinery — the SteeringVector container already supports multi-layer
  dicts, so no new hook code. Per-layer norms are reported via norms().
- MultiLayerLoRAObject: rank-r (A, B) per adapted module — by default (the
  paper's object) that layer's attention o_proj, B zero-init (exactly the
  unsteered model at step 0), the multi-layer twin of
  objects.RankOneLoRAObject; since the LoRA architecture search (FD 6,
  2026-09-29) also any module SET of extraction.lora_modules (q/k/v/o,
  the FFN's gate/up/down incl. a sparse-MoE's shared expert and its
  routed experts as joint 3-D per-expert factors), every set applied at
  every block that has the module; and a LeCun-scaled random init behind
  ``init_scheme`` (FD 7) with the paper's zero init as the default.

Position-masked (decode-only) fits (2026-08-20): both objects take the
hooks.PositionMaskedFit seam, mirroring their single-layer siblings
exactly. MultiLayerVectorObject mirrors the `vector` arm: under a
``set_fit_positions("response", H)`` leg its apply() routes the fit
positions into hooks.steering, whose live SteeringHandle both masks the
per-layer additions (SteeringHandle._position_mask — the ONE mask home)
and is yielded to the trainer for the per-row boundaries + the
loss_gather_masks tie. MultiLayerLoRAObject mirrors lora_r1
(objects.RankOneLoRAObject): one VECTORLESS tracker handle per apply()
ctx, shared by every layer's o_proj hook, each delta gated through
PositionMaskedFit._fit_masked (torch.where — masked positions stay
BIT-identical), the tracker yielded for the same mask→loss wiring.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from ..hooks import PositionMaskedFit, steering
from ..models import (attention_oproj, get_decoder_layers, hidden_size,
                      num_layers, oproj_layers)
from ..vectors import SteeringVector
from .lora_modules import (EXPERTS_DOWN, EXPERTS_GATE_UP, INIT_SCHEMES,
                           PAPER_LORA_MODULES, active_experts_of,
                           init_generator, layers_by_module, lecun_sigma,
                           normalise_modules,
                           per_expert_rank as per_expert_rank_rule,
                           resolve_targets, routed_rank_of, set_id_for,
                           targets_summary)


def _default_layers(model) -> list[int]:
    return list(range(num_layers(model)))


class MultiLayerVectorObject(PositionMaskedFit):
    """One additive residual d-vector per layer, one shared eval scale.

    Position-masked fits: the multi-layer twin of the single-layer `vector`
    arm, which is hooks.steering-backed — so the fit's mask leg is honored
    by ROUTING the fit positions into steering() itself. The steering
    handle is the mask source (SteeringHandle._position_mask, never a
    reimplementation) and apply() yields it, so the trainer sets per-row
    boundaries and ties the loss gather to the SAME mask
    (distill.loss_gather_masks). The default leg is byte-identical to the
    historical behavior (positions=self.positions, no header boundary).
    """

    kind = "vector_all"

    def __init__(self, model, layers: list[int] | None = None,
                 positions: str = "all", skip_bos: bool = True,
                 response_header_len: int | None = None):
        d = hidden_size(model)
        self.layers = list(layers) if layers is not None else _default_layers(model)
        self.params: dict[int, nn.Parameter] = {
            l: nn.Parameter(torch.zeros(d)) for l in self.layers
        }
        self.positions = positions
        self.skip_bos = skip_bos
        # Generation-time boundary for positions="response" (H = assistant-header
        # token length); set by the jailbreak defense factory, None otherwise.
        self.response_header_len = response_header_len

    @property
    def trainable(self) -> list[nn.Parameter]:
        return [self.params[l] for l in self.layers]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.trainable)

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def apply(self, model, scale: float = 1.0):
        # Inference-time leg (the shared primitive): positions /
        # response_header_len set on the OBJECT (the jailbreak defense
        # factory's positions="response").
        positions, header = self.positions, self.response_header_len
        if self.fit_positions != "all":
            # The fit's mask leg (PositionMaskedFit contract): steer only
            # the boundary-onward positions, header-steered when H is set.
            positions = self.fit_positions
            header = self.fit_response_header_len
        return steering(
            model,
            {l: self.params[l] for l in self.layers},
            scale=scale,
            positions=positions,
            skip_bos=self.skip_bos,
            response_header_len=header,
        )

    def norms(self) -> dict[int, float]:
        """Per-layer vector norms (the sidecar report of the plan)."""
        return {l: float(self.params[l].detach().norm()) for l in self.layers}

    def to_steering_vector(self, model_name: str = "",
                           instruction_id: str = "") -> SteeringVector:
        """The same intervention as a SteeringVector artifact (multi-layer
        dicts are already supported by the container and by hooks)."""
        return SteeringVector(
            vectors={l: self.params[l].detach().float().cpu()
                     for l in self.layers},
            method="vector_all",
            model_name=model_name,
            instruction_id=instruction_id,
            metadata={"positions": self.positions, "skip_bos": self.skip_bos,
                      "norms": {str(l): n for l, n in self.norms().items()}},
        )

    def to_(self, device) -> None:
        for p in self.params.values():
            p.data = p.data.to(device)

    def save(self, path: Path) -> None:
        torch.save(
            {"kind": self.kind,
             "vectors": {str(l): self.params[l].detach().float().cpu()
                         for l in self.layers},
             "positions": self.positions,
             "skip_bos": self.skip_bos,
             # The FIT's mask leg (PositionMaskedFit) — distinct keys, since
             # "positions" above is this class's serve-time DEPLOYMENT attr.
             # Part of the artifact's identity (DiffMeanObject/LoReFT
             # precedent, 2026-08-21): a response-fit vector_all previously
             # re-hydrated ALL-POSITION and would silently run the all-token
             # method under a decode-only label.
             "fit_positions": self.fit_positions,
             "fit_response_header_len": self.fit_response_header_len},
            path,
        )

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "MultiLayerVectorObject":
        layers = sorted(int(k) for k in blob["vectors"])
        obj = cls(model, layers=layers, positions=blob.get("positions", "all"),
                  skip_bos=blob.get("skip_bos", True))
        for l in layers:
            obj.params[l].data = blob["vectors"][str(l)].clone()
        # Backward-compatible fit-leg restore: no fit_positions key -> the
        # historical all-token fit, byte-identical behavior.
        if blob.get("fit_positions", "all") != "all":
            obj.set_fit_positions(blob["fit_positions"],
                                  blob.get("fit_response_header_len"))
        return obj


def _target_generator(seed: int, layer: int, module: str) -> torch.Generator:
    """One deterministic RNG stream per (seed, layer, module) for the
    GENERAL module-set object: the same seed and architecture give the
    same initial tensors in every process (the search trial, the
    production fit and the constitution/control twins of a seed all draw
    from these streams; tests pin it). The legacy o_proj object keeps its
    historical ``manual_seed(seed + layer)`` stream untouched."""
    return init_generator("lora_all", int(seed), int(layer), module)


def _grouped_linear():
    """transformers' grouped GEMM dispatcher (``F.grouped_mm`` /
    ``torch._grouped_mm`` with the fallback loop; 5.14.1 and 5.15.0 both
    ship it), or None on a transformers without it."""
    try:
        from transformers.integrations.moe import _grouped_linear as gl
    except Exception:  # noqa: BLE001 — an older/newer transformers
        return None
    return gl


#: grouped_mm requires 16-byte-aligned strides (bf16: 8 elements); the
#: per-expert LoRA rank r_e is 1 for ranks 4/16, so the factor's rank axis
#: is zero-PADDED to a multiple of this at forward time (padded slots are
#: exactly zero and contribute nothing; the trainable tensors keep the
#: true r_e so parameter counts are honest).
_EXPERT_RANK_PAD = 8


def _pad_rank(t: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
    """``t`` cast to ``dtype`` with its rank axis ``dim`` zero-padded to a
    multiple of :data:`_EXPERT_RANK_PAD` — cast FIRST (the padded slots are
    exact zeros in any dtype), one ``F.pad`` kernel instead of a zeros
    allocation plus a cat over the wider tensor."""
    t = t.to(dtype)
    pad = (-int(t.shape[dim])) % _EXPERT_RANK_PAD
    if pad == 0:
        return t
    spec = [0, 0] * (t.dim() - 1 - dim) + [0, pad]   # F.pad counts from the LAST dim
    return nn.functional.pad(t, spec)


class _ExpertLoRATerm:
    """One routed-expert LoRA contribution at ``scale``: the fused gate_up
    factors (``A_gu [E, r_e, d]``, ``B_gu [E, 2I, r_e]``) and/or the down
    factors (``A_d [E, r_e, I]``, ``B_d [E, d, r_e]``); either pair may be
    None (the ``down`` set has no gate_up factor)."""

    __slots__ = ("A_gu", "B_gu", "A_d", "B_d", "scale")

    def __init__(self, A_gu, B_gu, A_d, B_d, scale: float):
        self.A_gu, self.B_gu, self.A_d, self.B_d, self.scale = A_gu, B_gu, A_d, B_d, float(scale)


def _experts_lora_forward(experts: nn.Module, terms: list[_ExpertLoRATerm]):
    """A replacement ``forward`` for HF's 3-D experts module
    (``Qwen3_5MoeExperts`` and its siblings: ``gate_up_proj [E, 2I, d]``,
    ``down_proj [E, d, I]``, ``forward(hidden_states, top_k_index,
    top_k_weights)``) that computes the routed FFN WITH the LoRA terms in
    one pass. The routed-expert delta is not additive on the output (the
    gate_up LoRA changes the activation that feeds down_proj), so a
    post-hook cannot express it; the forward is swapped for the duration
    of ``apply()`` instead. Mirrors transformers' ``grouped_mm_experts_forward``
    (sort token-expert pairs by expert, one grouped GEMM per projection,
    the module's own ``_apply_gate``) and adds, per projection,
    ``scale · B_e (A_e x)`` through the same grouped GEMM; falls back to a
    per-expert loop on a transformers without the dispatcher. With every
    B at zero this equals the base forward (up to the GEMM path's own
    rounding), so an untrained object is the unsteered model."""
    E = int(experts.num_experts)
    gl = _grouped_linear()
    act = getattr(experts, "_apply_gate", None)
    transposed = bool(getattr(experts, "is_transposed", False))

    def gate(gu: torch.Tensor) -> torch.Tensor:
        if act is not None:
            return act(gu)
        g, u = gu.chunk(2, dim=-1)
        return experts.act_fn(g) * u

    def lora_delta(x_g, A3, B3, offs, scale):
        # x_g [S, in] -> (x A^T) [S, r_pad] -> (· B^T) [S, out]
        a = _pad_rank(A3, 1, x_g.dtype)
        b = _pad_rank(B3, 2, x_g.dtype)
        h = gl(x_g, a, offs, is_transposed=False)
        return gl(h, b, offs, is_transposed=False) * scale

    def forward_grouped(hidden_states, top_k_index, top_k_weights):
        device = hidden_states.device
        k = top_k_index.size(-1)
        n_tok = hidden_states.size(0)
        d = hidden_states.size(-1)
        weights = top_k_weights.reshape(-1)
        ids = top_k_index.reshape(-1)
        ids_g, perm = torch.sort(ids)
        x_g = hidden_states[perm // k]
        w_g = weights[perm]
        histc_in = ids_g.float() if device.type in ("cpu", "mps") else ids_g.int()
        per_expert = torch.histc(histc_in, bins=E, min=0, max=E - 1)
        offs = torch.cumsum(per_expert, dim=0, dtype=torch.int32)
        sentinel = (ids_g >= E).unsqueeze(-1)
        ids_g.clamp_(max=E - 1)
        x_g = x_g.masked_fill(sentinel, 0.0)
        W_gu = experts.gate_up_proj
        gu = gl(x_g, W_gu, offs, is_transposed=transposed)
        for t in terms:
            if t.A_gu is not None:
                gu = gu + lora_delta(x_g, t.A_gu, t.B_gu, offs, t.scale)
        h = gate(gu)
        W_d = experts.down_proj
        out = gl(h, W_d, offs, is_transposed=transposed)
        for t in terms:
            if t.A_d is not None:
                out = out + lora_delta(h, t.A_d, t.B_d, offs, t.scale)
        out = out * w_g.unsqueeze(-1)
        out = out.masked_fill(sentinel, 0.0)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.size(0), device=device)
        out = out[inv]
        return out.view(n_tok, k, d).sum(dim=1).to(hidden_states.dtype)

    def forward_loop(hidden_states, top_k_index, top_k_weights):
        # The eager reference: per hit expert, gate_up -> gate -> down,
        # each with its LoRA term (the same arithmetic as the grouped
        # path; used when transformers has no grouped dispatcher and by
        # the tests as the ground truth).
        out = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e_t in hit:
            e = int(e_t[0])
            pos, tok = torch.where(mask[e])
            x = hidden_states[tok]
            gu = nn.functional.linear(x, experts.gate_up_proj[e])
            for t in terms:
                if t.A_gu is not None:
                    gu = gu + t.scale * ((x @ t.A_gu[e].to(x.dtype).T) @ t.B_gu[e].to(x.dtype).T)
            h = gate(gu)
            y = nn.functional.linear(h, experts.down_proj[e])
            for t in terms:
                if t.A_d is not None:
                    y = y + t.scale * ((h @ t.A_d[e].to(h.dtype).T) @ t.B_d[e].to(h.dtype).T)
            y = y * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, y.to(out.dtype))
        return out

    return forward_grouped if gl is not None else forward_loop


def _add_expert_overrides(handles: list, mask_handle, experts: dict,
                          terms_by_layer: dict) -> None:
    """Append one :class:`_ForwardOverride` per MoE block in ``terms_by_layer``
    (``{layer: [term, ...]}``) onto ``handles``. A position-masked fit is
    REFUSED for routed-expert targets (the experts module sees a
    token-flattened batch and cannot apply a per-position mask; every
    corpus fit of the pipeline is an all-position fit) — the linear hooks
    already registered are removed first so the refusal leaves the model
    clean."""
    if not terms_by_layer:
        return
    if mask_handle is not None:
        for h in handles:
            h.remove()
        raise NotImplementedError(
            "position-masked fits are not supported for routed-expert "
            "LoRA targets (the experts module sees a token-flattened "
            "batch); the corpus fits are all-position fits")
    for l, terms in terms_by_layer.items():
        handles.append(_ForwardOverride(experts[l], _experts_lora_forward(experts[l], terms)))


def _lora_hook(A: nn.Parameter, B: nn.Parameter, scale: float, fit_masked):
    """The rank-r forward hook every 2-D linear target runs: ``delta = (x @
    A) @ B.T`` (A ``[d_in, r]``, B ``[d_out, r]``), added to the module's
    output at ``scale`` and routed through the object's ``_fit_masked``
    (exact position masking under a tracker handle). One body for the
    legacy o_proj object at rank > 1 and for every general-mode linear
    target."""
    def hook(_mod, inputs, output):
        x = inputs[0]
        coef = x.to(A.dtype) @ A  # [..., r]
        delta = coef @ B.T  # [..., d_out]
        steered = output + (scale * delta).to(output.dtype)
        return fit_masked(steered, output)

    return hook


class _ForwardOverride:
    """A removable handle (``.remove()``) that swaps one module's bound
    ``forward`` for the duration of an ``apply()`` context — the
    routed-expert analogue of a ``register_forward_hook`` handle."""

    def __init__(self, module: nn.Module, forward):
        self._module = module
        if "forward" in module.__dict__:
            raise RuntimeError(
                f"{type(module).__name__} already carries an instance-level "
                "forward override — two LoRA objects cannot both patch one "
                "experts module; compose their terms into ONE override "
                "(LoRASubtractionObject does)")
        module.forward = forward

    def remove(self) -> None:
        if "forward" in self._module.__dict__:
            del self._module.forward


class MultiLayerLoRAObject(PositionMaskedFit):
    """Rank-r LoRA (dW = B A^T) on a MODULE SET applied at every eligible
    decoder block.

    The paper's object (and the default) is ``modules=("o_proj",)``: one
    (A_l, B_l) per full-attention block's ``o_proj``, exactly the class's
    historical shape — the LEGACY mode below, bit-identical to the
    pre-module-set code (same RNG streams, same op sequence, same blob),
    so the paper's 64 LoRA objects load and serve unchanged. Any other set
    (``qkvo``, ``attn_mlp``, ``mlp``, ``down``) is the
    GENERAL mode: targets are resolved by :func:`lora_modules.resolve_targets`
    (a capability probe; every block that HAS the module), factors are
    keyed ``"<layer>.<module>"``, the routed experts of a sparse-MoE FFN
    carry the joint 3-D factors of "LoRA Without Regret" (uniform
    per-expert rank ``max(1, r // active)``, full rank on the shared
    expert) and are applied by swapping the experts module's forward
    (:func:`_experts_lora_forward`), while every linear target keeps the
    forward-hook delta ``scale · (x @ A) @ B.T``.

    Initialisation (FD 7): A ~ N(0, 1/d_in) always; B = 0 under the paper's
    default ``init_scheme="zero"`` (the untrained object IS the unsteered
    model), or B ~ N(0, σ²) with σ = 1e-3/√r under ``"random_lecun_1e-3"``
    (FD 7's run-override scheme). Draws come from the fit seed; two
    objects built with the same seed and architecture are identical, so
    the constitution and control twins of a seed share their init.

    ``rank`` (twosi re-tune@32, 2026-08-27: LoRA rank joined the lora arm's
    search grid {1, 4, 16}) is per-object. rank=1 in legacy mode keeps the
    LEGACY parameter shapes (1-D A_l [d_in], B_l [d_out]) and the exact
    legacy op sequence (``coef.unsqueeze(-1) * B``); rank>1 uses 2-D
    parameters (A_l [d_in, r], B_l [d_out, r]; delta = (x @ A) @ B^T) and
    saves with an explicit ``rank`` key; load_blob infers rank from the
    stored tensors' ndim, so legacy rank-1 blobs re-hydrate byte-identically.

    Position-masked fits (PositionMaskedFit): every linear target keeps
    RankOneLoRAObject's wiring — a position-wise linear map, so
    restricting the delta to the handle's mask is EXACT (masked positions
    BIT-identical via torch.where), one vectorless tracker handle per
    apply() shared by every hook. A masked fit is REFUSED for routed-expert
    targets (the experts module sees a token-flattened batch and cannot
    apply a per-position mask); every corpus fit of the pipeline is an
    all-position fit.
    """

    kind = "lora_all"

    def __init__(self, model, layers: list[int] | None = None, seed: int = 0,
                 rank: int = 1, *, modules=("o_proj",),
                 per_expert_rank: int | dict | None = None,
                 init_scheme: str = "zero"):
        rank = int(rank)
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank!r}")
        if init_scheme not in INIT_SCHEMES:
            raise ValueError(
                f"init_scheme must be one of {INIT_SCHEMES}, got {init_scheme!r}")
        self.rank = rank
        self.seed = int(seed)
        self.init_scheme = str(init_scheme)
        self.modules = normalise_modules(modules)
        self.set_id = set_id_for(self.modules)
        self.legacy = self.modules == PAPER_LORA_MODULES
        self.A: dict = {}
        self.B: dict = {}
        self._o_projs: dict[int, nn.Module] = {}
        self._linear: dict[str, nn.Module] = {}
        self._experts: dict[int, nn.Module] = {}
        if self.legacy:
            if per_expert_rank is not None:
                raise ValueError(
                    "per_expert_rank has no meaning for the o_proj-only object "
                    "(no routed experts are adapted)")
            self.per_expert_rank = None
            self._init_legacy(model, layers)
            return
        self._init_general(model, layers, per_expert_rank)

    # -- construction ---------------------------------------------------------

    def _init_legacy(self, model, layers) -> None:
        # CAPABILITY-FILTERED layer set: this object edits the
        # attention o_proj, so "every layer" means every layer that HAS one
        # (models.attention_oproj — self_attn.o_proj, or the NemotronH
        # attention mixer's o_proj). On hybrid models the default is the
        # attention subset — Qwen3.5's qwen3_5: 16 of 64 layers (the rest
        # linear_attention with no self_attn); Nemotron-3-Nano-4B: 4 of 42 (12, 17, 24, 32; the rest Mamba2/MLP
        # mixers) — where the old range(num_layers) default died with an
        # AttributeError before training; on all-full-attention models
        # (Llama, gemma-4) the filtered default is the same all-layer set as
        # before, bit-for-bit.
        capable = oproj_layers(model)
        if layers is None:
            self.layers = capable
            if not self.layers:
                raise ValueError(
                    "no decoder layer on this model has self_attn.o_proj (or "
                    "an attention-mixer o_proj) — the all-layer rank-1 o_proj "
                    "LoRA has nothing to target (refusing to construct a "
                    "silent no-op object)")
        else:
            # An EXPLICIT layer list is the caller's claim; a member without
            # o_proj is a caller error — fail loudly (repo rule), never
            # silently narrow what was asked for.
            self.layers = list(layers)
            missing = sorted(set(self.layers) - set(capable))
            if missing:
                raise ValueError(
                    f"layers {missing} have no self_attn.o_proj (nor an "
                    "attention-mixer o_proj; hybrid model? linear-attention, "
                    "Mamba and MLP blocks carry no attention output "
                    "projection) — the rank-1 o_proj LoRA can only target "
                    f"{capable}")
        rank = self.rank
        decoder = get_decoder_layers(model)
        sigma = lecun_sigma(rank) if self.init_scheme != "zero" else None
        for l in self.layers:
            g = torch.Generator().manual_seed(self.seed + l)
            # The SAME resolver the capability filter above used, so the
            # module attached is by construction the one whose presence
            # admitted the layer (self_attn.o_proj, or NemotronH's
            # mixer.o_proj) — never a second spelling that could disagree.
            oproj = attention_oproj(decoder[l])
            # A dots the o_proj INPUT, B is added to its OUTPUT — different
            # dims on non-square-o_proj models (gemma-2-9b 4096->3584;
            # Nemotron-3-Nano-4B 5120 = 40 heads x 128 -> hidden 3136);
            # sizing from hidden_size only works on Llama-like square o_proj.
            d_in = oproj.weight.shape[1]
            d_out = oproj.weight.shape[0]
            if rank == 1:
                # LEGACY shapes + RNG draw order: bit-identical to the
                # pre-rank object (same generator, same randn(d_in) call).
                self.A[l] = nn.Parameter(
                    torch.randn(d_in, generator=g) / d_in**0.5)
                b = torch.zeros(d_out)
                if sigma is not None:
                    # The B draw follows A on the SAME stream (a second
                    # randn on g) — the zero default never touches g, so
                    # zero-init objects are bit-identical to before.
                    b = torch.randn(d_out, generator=g) * sigma
                self.B[l] = nn.Parameter(b)
            else:
                # Per-column init scale matches rank-1's (each column
                # ~randn(d_in)/sqrt(d_in)); B zero-init keeps the untrained
                # object exactly the unsteered model at every rank.
                self.A[l] = nn.Parameter(
                    torch.randn(d_in, rank, generator=g) / d_in**0.5)
                b = torch.zeros(d_out, rank)
                if sigma is not None:
                    b = torch.randn(d_out, rank, generator=g) * sigma
                self.B[l] = nn.Parameter(b)
            self._o_projs[l] = oproj
        # The target view of the legacy object (the export and the registry
        # read sites/layers from here); resolve_targets on an explicit
        # o_proj layer list is exactly the capability check above.
        self.targets = resolve_targets(model, ("o_proj",), rank=rank,
                                       layers=self.layers)

    def _init_general(self, model, layers, per_expert_rank) -> None:
        routed = routed_rank_of(per_expert_rank, self.rank)
        self.targets = resolve_targets(model, self.modules, rank=self.rank,
                                       layers=layers, per_expert_rank=routed)
        self.layers = sorted({t.layer for t in self.targets})
        experts_targets = [t for t in self.targets if t.kind == "experts"]
        if experts_targets:
            r_e = int(experts_targets[0].rank)
            active = active_experts_of(model)
            if routed is not None and active is not None and \
                    int(routed) != per_expert_rank_rule(self.rank, active):
                raise ValueError(
                    f"per_expert_rank={routed} contradicts FD 6's rule "
                    f"max(1, {self.rank} // {active}) = "
                    f"{per_expert_rank_rule(self.rank, active)}")
            self.per_expert_rank = {"shared_expert": self.rank, "routed_expert": r_e}
        else:
            self.per_expert_rank = None
        decoder = get_decoder_layers(model)
        sigma_lin = lecun_sigma(self.rank) if self.init_scheme != "zero" else None
        for t in self.targets:
            g = _target_generator(self.seed, t.layer, t.module)
            host = decoder[t.layer]
            for attr in t.site.split(".")[:-1]:
                host = getattr(host, attr)
            if t.kind == "linear":
                mod = getattr(host, t.site.split(".")[-1])
                d_in, d_out = t.in_features, t.out_features
                self.A[t.key] = nn.Parameter(torch.randn(d_in, self.rank, generator=g) / d_in**0.5)
                b = torch.zeros(d_out, self.rank)
                if sigma_lin is not None:
                    b = torch.randn(d_out, self.rank, generator=g) * sigma_lin
                self.B[t.key] = nn.Parameter(b)
                self._linear[t.key] = mod
                if t.module == "o_proj":
                    self._o_projs[t.layer] = mod
            else:
                E, r_e = int(t.num_experts), int(t.rank)
                d_in, d_out = t.in_features, t.out_features
                self.A[t.key] = nn.Parameter(torch.randn(E, r_e, d_in, generator=g) / d_in**0.5)
                b = torch.zeros(E, d_out, r_e)
                if self.init_scheme != "zero":
                    b = torch.randn(E, d_out, r_e, generator=g) * lecun_sigma(r_e)
                self.B[t.key] = nn.Parameter(b)
                self._experts[t.layer] = host  # the experts module itself

    # -- protocol -------------------------------------------------------------

    @property
    def factor_keys(self) -> list:
        """The ``A``/``B`` dict keys in target order: the layer ints of the
        legacy object, ``"<layer>.<module>"`` in general mode."""
        return list(self.layers) if self.legacy else [t.key for t in self.targets]

    @property
    def trainable(self) -> list[nn.Parameter]:
        keys = self.factor_keys
        return [self.A[k] for k in keys] + [self.B[k] for k in keys]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.trainable)

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def describe(self) -> dict:
        """The architecture facts the registry and the HPO table carry
        (``params_json`` names): rank, sorted modules, set id,
        per-expert rank map, init scheme, trainable count, adapted layers
        per module."""
        return {
            "object_kind": "lora",
            "lora_rank": int(self.rank),
            "lora_modules": list(self.modules),
            "lora_set_id": self.set_id,
            "per_expert_rank": (dict(self.per_expert_rank)
                                if self.per_expert_rank is not None else None),
            "init_scheme": self.init_scheme,
            "trainable_params": int(self.n_params()),
            "layers_by_module": layers_by_module(self.targets),
        }

    def _make_hook(self, layer: int, scale: float, mask_handle):
        """The LEGACY o_proj hook (rank-1 op sequence preserved bit-exactly)."""
        A, B = self.A[layer], self.B[layer]
        if self.rank != 1:
            return self._make_linear_hook(layer, scale, mask_handle)

        # The exact legacy op sequence (bit-identity with the pre-rank
        # object; a [d,1] matmul kernel is not guaranteed bitwise-equal
        # to the matvec, so rank 1 keeps the original path).
        def hook(_mod, inputs, output):
            x = inputs[0]
            coef = x.to(A.dtype) @ A  # [...,]
            delta = coef.unsqueeze(-1) * B  # [..., d]
            steered = output + (scale * delta).to(output.dtype)
            return self._fit_masked(mask_handle, steered, output)

        return hook

    def _make_linear_hook(self, key, scale: float, mask_handle):
        """The rank-r hook of one 2-D factor pair (``key`` a legacy layer int
        or a general ``"<layer>.<module>"``): :func:`_lora_hook` under this
        object's position mask."""
        return _lora_hook(self.A[key], self.B[key], scale,
                          lambda steered, output: self._fit_masked(mask_handle, steered, output))

    def expert_terms(self, scale: float) -> dict[int, _ExpertLoRATerm]:
        """Per MoE block, this object's routed-expert LoRA term at ``scale``
        (empty for objects without expert targets). LoRASubtractionObject
        composes two objects' terms into ONE forward override per block."""
        out: dict[int, _ExpertLoRATerm] = {}
        for l in self._experts:
            gu = f"{l}.{EXPERTS_GATE_UP}"
            dn = f"{l}.{EXPERTS_DOWN}"
            out[l] = _ExpertLoRATerm(
                self.A.get(gu), self.B.get(gu), self.A.get(dn), self.B.get(dn), scale)
        return out

    def register_linear_hooks(self, scale: float, mask_handle) -> list:
        """The forward hooks of every LINEAR target at ``scale`` (the
        legacy o_proj hooks in legacy mode)."""
        if self.legacy:
            return [
                self._o_projs[l].register_forward_hook(
                    self._make_hook(l, scale, mask_handle))
                for l in self.layers
            ]
        return [mod.register_forward_hook(self._make_linear_hook(key, scale, mask_handle))
                for key, mod in self._linear.items()]

    def register_hooks(self, scale: float, mask_handle) -> list:
        """Register this object's per-target hooks at ``scale`` under a
        caller-owned position tracker (``_fit_mask_handle``'s handle, or
        None); returns removable handles, which the caller removes. apply()
        is the one-object case; LoRASubtractionObject composes TWO objects
        under ONE tracker through ``register_linear_hooks`` +
        ``expert_terms``."""
        handles = self.register_linear_hooks(scale, mask_handle)
        _add_expert_overrides(handles, mask_handle, self._experts,
                              {l: [term] for l, term in self.expert_terms(scale).items()})
        return handles

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mask_handle = self._fit_mask_handle(model)
        handles = self.register_hooks(scale, mask_handle)
        try:
            yield mask_handle
        finally:
            for h in handles:
                h.remove()
            if mask_handle is not None:
                mask_handle.remove()

    def to_(self, device) -> None:
        for group in (self.A, self.B):
            for p in group.values():
                p.data = p.data.to(device)

    # -- artifact -------------------------------------------------------------

    def to_blob(self) -> dict:
        """The ``lora_all`` artifact dict save() persists (the ONE owner of
        that schema; LoRASubtractionObject composes derived artifacts from
        its parents' blobs through it). Legacy shape for the o_proj-only
        object (plus ``init_scheme`` only when it is not the zero default);
        the general shape adds ``lora_modules``, ``lora_set_id``,
        ``per_expert_rank``, ``init_scheme`` and ``targets`` and keys the
        factors ``"<layer>.<module>"``."""
        # Insertion order IS the pickled byte order (torch.save), so the legacy
        # key sequence below is the historical one exactly and the general
        # shape's five extra keys sit between ``rank`` and ``A`` as the
        # contract writes them.
        blob: dict = {
            "kind": self.kind,
            # The TARGETED layer set, explicit: on a hybrid-attention
            # model this is the capability-filtered subset (Qwen3.5-27B:
            # the 16 full-attention layers, not 64), so the artifact is
            # auditable without decoding the A/B key sets.
            "layers": [int(l) for l in self.layers],
            # rank rides explicitly (re-tune@32 search grid); rank-1 blobs
            # keep the legacy 1-D tensors, so pre-rank readers/artifacts
            # stay byte-compatible in both directions.
            "rank": int(self.rank),
        }
        if not self.legacy:
            blob.update({
                "lora_modules": list(self.modules),
                "lora_set_id": self.set_id,
                "per_expert_rank": (dict(self.per_expert_rank)
                                    if self.per_expert_rank is not None else None),
                "init_scheme": self.init_scheme,
                "targets": targets_summary(self.targets),
            })
        keys = self.factor_keys
        blob.update({
            "A": {str(k): self.A[k].detach().float().cpu() for k in keys},
            "B": {str(k): self.B[k].detach().float().cpu() for k in keys},
            # The FIT's mask leg is part of the artifact's identity (the
            # DiffMeanObject/LoReFT precedent, 2026-08-21; RankOneLoRAObject
            # persists the same two keys).
            "positions": self.fit_positions,
            "response_header_len": self.fit_response_header_len,
        })
        if self.legacy and self.init_scheme != "zero":
            blob["init_scheme"] = self.init_scheme
        return blob

    def save(self, path: Path) -> None:
        torch.save(self.to_blob(), path)

    @staticmethod
    def blob_is_general(blob: dict) -> bool:
        """True for the module-set (general) blob shape — a ``targets`` key
        — False for the legacy o_proj-only shape."""
        return isinstance(blob, dict) and "targets" in blob

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "MultiLayerLoRAObject":
        if cls.blob_is_general(blob):
            return cls._load_general(blob, model)
        layers = sorted(int(k) for k in blob["A"])
        if "layers" in blob and sorted(int(x) for x in blob["layers"]) != layers:
            raise ValueError(
                f"lora_all artifact: the blob's `layers` metadata "
                f"{sorted(int(x) for x in blob['layers'])} disagrees with "
                f"the layers its A tensors actually cover {layers} — "
                "refusing to load a self-inconsistent artifact")
        # Legacy blobs (pre-rank) carry no "rank" key and 1-D tensors —
        # infer from the data so the key can never contradict the shapes.
        a0 = blob["A"][str(layers[0])]
        inferred = 1 if a0.dim() == 1 else int(a0.shape[1])
        stored = int(blob.get("rank", inferred))
        if stored != inferred:
            raise ValueError(
                f"lora_all blob is internally inconsistent: rank key says "
                f"{stored} but the stored A tensors are rank {inferred}")
        obj = cls(model, layers=layers, rank=inferred,
                  init_scheme=str(blob.get("init_scheme", "zero")))
        for l in layers:
            A, B = blob["A"][str(l)], blob["B"][str(l)]
            # Accept the two shapes the kind ships in (see the class
            # docstring): 1-D rank-1 vectors (this repo's trainer) or 2-D
            # [d, k] rank-k matrices. Anything the
            # pair cannot agree on refuses HERE — a mismatched pair used
            # to load silently and crash with a broadcast error inside
            # the o_proj hook at the first forward.
            if A.dim() != B.dim() or A.dim() not in (1, 2):
                raise ValueError(
                    f"lora_all artifact layer {l}: A is "
                    f"{tuple(A.shape)} and B is {tuple(B.shape)} — the "
                    "pair must both be 1-D (rank-1 vectors) or both "
                    "[d, k] matrices; refusing")
            rank = 1 if A.dim() == 1 else int(A.shape[1])
            if A.dim() == 2 and int(B.shape[1]) != rank:
                raise ValueError(
                    f"lora_all artifact layer {l}: A rank {rank} != B "
                    f"rank {int(B.shape[1])} (A {tuple(A.shape)}, B "
                    f"{tuple(B.shape)}) — refusing")
            if "rank" in blob and int(blob["rank"]) != rank:
                raise ValueError(
                    f"lora_all artifact layer {l}: tensors are rank "
                    f"{rank} but the blob's `rank` metadata says "
                    f"{int(blob['rank'])} — the artifact's own records "
                    "disagree; refusing")
            obj.A[l].data = A.clone()
            obj.B[l].data = B.clone()
        # Backward-compatible fit-leg restore (the DiffMeanObject pattern):
        # no positions key -> the historical all-token fit, byte-identical.
        if blob.get("positions", "all") != "all":
            obj.set_fit_positions(blob["positions"],
                                  blob.get("response_header_len"))
        return obj

    @classmethod
    def _load_general(cls, blob: dict, model) -> "MultiLayerLoRAObject":
        for k in ("lora_modules", "rank", "targets", "A", "B", "layers"):
            if k not in blob:
                raise ValueError(f"lora_all (module-set) artifact lacks {k!r}; refusing")
        per = blob.get("per_expert_rank")
        obj = cls(model, layers=[int(l) for l in blob["layers"]], rank=int(blob["rank"]),
                  modules=blob["lora_modules"],
                  per_expert_rank=(dict(per) if isinstance(per, dict) else per),
                  init_scheme=str(blob.get("init_scheme", "zero")))
        if obj.legacy:
            raise ValueError(
                "lora_all artifact carries a `targets` block but names the "
                "o_proj-only set — that is the legacy shape; refusing the mixed form")
        want = [t.key for t in obj.targets]
        have = sorted(blob["A"])
        if sorted(want) != have or sorted(blob["B"]) != have:
            raise ValueError(
                "lora_all (module-set) artifact: its factor keys "
                f"{have[:6]}{'…' if len(have) > 6 else ''} do not match the "
                f"{len(want)} targets the set {list(obj.modules)} resolves to on "
                f"{type(model).__name__} ({sorted(set(want) ^ set(have))[:6]}) — "
                "refusing to load against a different architecture")
        recorded = targets_summary(obj.targets)
        for m, rec in blob["targets"].items():
            mine = recorded.get(m)
            if mine is None or mine["site"] != rec.get("site") or \
                    sorted(mine["layers"]) != sorted(int(x) for x in rec.get("layers", [])) or \
                    int(mine["rank"]) != int(rec.get("rank", -1)):
                raise ValueError(
                    f"lora_all (module-set) artifact: target {m!r} recorded as "
                    f"{rec} but resolves to {mine} here; refusing")
        for key in want:
            A, B = blob["A"][key], blob["B"][key]
            if tuple(A.shape) != tuple(obj.A[key].shape) or tuple(B.shape) != tuple(obj.B[key].shape):
                raise ValueError(
                    f"lora_all (module-set) artifact {key}: A {tuple(A.shape)} / B "
                    f"{tuple(B.shape)} vs expected {tuple(obj.A[key].shape)} / "
                    f"{tuple(obj.B[key].shape)}; refusing")
            obj.A[key].data = A.clone()
            obj.B[key].data = B.clone()
        if blob.get("positions", "all") != "all":
            obj.set_fit_positions(blob["positions"], blob.get("response_header_len"))
        return obj


#: The general-shape keys a lora_all / lora_subtraction blob carries beyond
#: the legacy ones (the export and the subtraction builder copy them).
GENERAL_BLOB_KEYS = ("lora_modules", "lora_set_id", "per_expert_rank",
                     "init_scheme", "targets")


class LoRASubtractionObject(PositionMaskedFit):
    """Task-vector DIFFERENCE of two same-shape lora_all fits:
    ΔW(pos) − ΔW(neg), per target (sdf-base-steering round-2, 2026-09-02:
    pos = a constv2 mix-cell ``lora_r4_ce`` fit, neg = the register-matched
    control ``constv2ctl-100k__lora_r4_ce`` fit — the values-specific
    remainder as a SERVABLE object rather than a paired probe statistic).

    Representation is STRUCTURAL, never a re-factorization: the artifact
    stores both parents' (A, B) factor sets bit-exact, and apply()
    registers each parent's OWN hook per linear target — the positive
    parent's at ``+scale``, the negative's at ``-scale`` — on the same
    module. Forward hooks compose additively (each sees the previous hook's
    output; the module's INPUT is untouched by either), so the effective
    edit is exactly scale · (Δ_pos(x) − Δ_neg(x)), with the positive term
    computed by the byte-identical op sequence the mix fit itself serves
    under. That structure is what makes the algebra pins exact: dropping
    the negative term recovers the corpus (mix) object's deltas BITWISE
    (a dense ΔW_mix − ΔW_ctl subtraction could never be re-added to
    bitwise identity in floating point). Routed-expert targets compose
    the two parents' terms into ONE experts-forward override per block
    (``+scale`` and ``−scale`` terms; the same arithmetic). Served natively
    as the rank-2r concatenation ``A = [A_pos; A_neg]``, ``B = [B_pos, −B_neg]``
    whatever the module set (the export owns that layout).

    Derived and eval-only: ``trainable`` is empty (nothing here may be
    trained further — refit the parents instead), provenance (parent cell
    ids + object shas) rides inside the blob, and the artifact lives in a
    ``derived/`` namespace, NEVER the fits banks (scripts/sdfbase/
    lora_subtract.py is the one builder)."""

    kind = "lora_subtraction"

    def __init__(self, pos: MultiLayerLoRAObject, neg: MultiLayerLoRAObject,
                 provenance: dict | None = None):
        self._refuse_mismatch(
            (pos.layers, neg.layers), (pos.rank, neg.rank),
            tuple({(k, t): tuple(getattr(p, t)[k].shape)
                   for t in ("A", "B") for k in getattr(p, t)} for p in (pos, neg)),
            ((pos.fit_positions, pos.fit_response_header_len),
             (neg.fit_positions, neg.fit_response_header_len)),
            (pos.describe(), neg.describe()))
        for l in pos.layers:
            if l in pos._o_projs and pos._o_projs[l] is not neg._o_projs.get(l):
                raise RuntimeError(
                    f"lora_subtraction: layer {l} parents hold different "
                    "o_proj modules — both must be hydrated against the "
                    "SAME model")
        for k, m in pos._linear.items():
            if neg._linear.get(k) is not m:
                raise RuntimeError(
                    f"lora_subtraction: target {k} parents hold different "
                    "modules — both must be hydrated against the SAME model")
        for l, m in pos._experts.items():
            if neg._experts.get(l) is not m:
                raise RuntimeError(
                    f"lora_subtraction: layer {l} parents hold different "
                    "experts modules — both must be hydrated against the SAME model")
        self.pos = pos
        self.neg = neg
        self.layers = list(pos.layers)
        self.rank = int(pos.rank)
        self.modules = pos.modules
        self.set_id = pos.set_id
        self.per_expert_rank = pos.per_expert_rank
        self.provenance = dict(provenance or {})
        if pos.fit_positions != "all":
            self.set_fit_positions(pos.fit_positions,
                                   pos.fit_response_header_len)

    @staticmethod
    def _refuse_mismatch(layers: tuple, ranks: tuple, shapes: tuple,
                         fits: tuple, archs: tuple | None = None) -> None:
        """The same-shape preconditions, each a (pos, neg) pair read off
        either hydrated parents (``__init__``) or raw lora_all blobs
        (``compose_blob``): equal layer sets, equal rank, equal (A, B)
        shapes per factor key (``shapes`` maps (key, "A"|"B") -> shape),
        the same fit-position leg, and — for the general shape — the same
        module set / per-expert rank / init scheme. ONE wording for both
        entry points."""
        if list(layers[0]) != list(layers[1]):
            raise ValueError(
                f"lora_subtraction: parent layer sets differ "
                f"({layers[0]} vs {layers[1]}) — the task-vector "
                "difference is only defined layerwise on the same set")
        if ranks[0] != ranks[1]:
            raise ValueError(
                f"lora_subtraction: parent ranks differ ({ranks[0]} vs "
                f"{ranks[1]}) — same-shape parents only")
        if set(shapes[0]) != set(shapes[1]):
            raise ValueError(
                "lora_subtraction: parents adapt different targets "
                f"({sorted(set(shapes[0]) ^ set(shapes[1]))[:6]}) — same-shape parents only")
        for (k, tag), a in shapes[0].items():
            b = shapes[1][(k, tag)]
            if a != b:
                raise ValueError(
                    f"lora_subtraction: target {k} {tag} shapes differ "
                    f"({a} vs {b})")
        if fits[0] != fits[1]:
            raise ValueError(
                "lora_subtraction: parents carry different fit-position "
                f"legs {fits} — subtracting an all-position fit from a "
                "masked one is not a defined task vector")
        if archs is not None:
            for key in ("lora_modules", "per_expert_rank", "init_scheme"):
                if key not in archs[0] and key not in archs[1]:
                    continue
                if archs[0].get(key) != archs[1].get(key):
                    raise ValueError(
                        f"lora_subtraction: parents differ in {key} "
                        f"({archs[0].get(key)} vs {archs[1].get(key)}) — the "
                        "difference is defined only between architecture twins")

    @property
    def trainable(self) -> list[nn.Parameter]:
        return []  # derived object — nothing to train, honestly empty

    def n_params(self) -> int:
        return 0

    def inject(self, rendered_prompt: str) -> str:
        return rendered_prompt

    def describe(self) -> dict:
        d = self.pos.describe()
        d.update({"object_kind": "lora_subtraction",
                  "trainable_params": 0,
                  "parent_trainable_params": int(self.pos.n_params())})
        return d

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        mask_handle = self._fit_mask_handle(model)
        # Per linear target, registration order = composition order: the +
        # parent's delta, then the − parent's, each through
        # MultiLayerLoRAObject's OWN hook (the bit-identity anchor for the
        # algebra pins), all under ONE shared position tracker.
        handles = (self.pos.register_linear_hooks(scale, mask_handle)
                   + self.neg.register_linear_hooks(-scale, mask_handle))
        neg_terms = self.neg.expert_terms(-scale)
        _add_expert_overrides(handles, mask_handle, self.pos._experts,
                              {l: [term, neg_terms[l]]
                               for l, term in self.pos.expert_terms(scale).items()})
        try:
            yield mask_handle
        finally:
            for h in handles:
                h.remove()
            if mask_handle is not None:
                mask_handle.remove()

    def to_(self, device) -> None:
        self.pos.to_(device)
        self.neg.to_(device)

    def save(self, path: Path) -> None:
        # compose_blob is the ONE owner of the artifact schema (validated);
        # the parents serialize through their own to_blob(). Honest
        # provenance rides IN the artifact: parent cell ids + parent
        # object.pt sha256s + model key (lora_subtract.py fills these; an
        # empty dict marks an ad-hoc/test object).
        torch.save(self.compose_blob(self.pos.to_blob(), self.neg.to_blob(),
                                     provenance=self.provenance), path)

    @staticmethod
    def _parent_blob(blob: dict, side: str) -> dict:
        """One parent's lora_all blob, lifted back out of the artifact
        (MultiLayerLoRAObject.load_blob treats positions "all" as the
        historical all-token fit, so the leg keys ride unconditionally)."""
        parent = {"kind": "lora_all",
                  "layers": blob["layers"],
                  "rank": blob["rank"],
                  "A": blob[f"A_{side}"],
                  "B": blob[f"B_{side}"],
                  "positions": blob.get("positions", "all"),
                  "response_header_len": blob.get("response_header_len")}
        for k in GENERAL_BLOB_KEYS:
            if k in blob:
                parent[k] = blob[k]
        return parent

    @classmethod
    def compose_blob(cls, pos_blob: dict, neg_blob: dict,
                     provenance: dict | None = None) -> dict:
        """The subtraction ARTIFACT from two saved lora_all blobs — pure
        tensor bookkeeping, NO model needed (the builder script runs on
        CPU without touching the base weights). Parent tensors ride into
        the artifact bit-exact (``.clone()`` only); the same-shape
        preconditions ``__init__`` enforces on hydrated parents are
        enforced here on the blobs (``_refuse_mismatch``, one wording),
        and ``load_blob`` re-validates through MultiLayerLoRAObject's own
        loader at hydration time. General-shape parents (``targets``) carry
        their architecture keys into the artifact; legacy parents produce
        the legacy artifact byte-for-byte."""
        parents = (pos_blob, neg_blob)
        for tag, b in zip(("pos", "neg"), parents):
            if b.get("kind") != "lora_all":
                raise ValueError(
                    f"lora_subtraction: {tag} parent kind is "
                    f"{b.get('kind')!r}, not 'lora_all'")
        general = tuple(MultiLayerLoRAObject.blob_is_general(b) for b in parents)
        if general[0] != general[1]:
            raise ValueError(
                "lora_subtraction: one parent is a module-set (general) "
                "artifact and the other the o_proj-only shape — not twins")
        if general[0]:
            key_sets = tuple(sorted(b["A"]) for b in parents)
            layer_sets = tuple(sorted(int(x) for x in b["layers"]) for b in parents)
            ranks = tuple(int(b["rank"]) for b in parents)
            archs = tuple({k: b.get(k) for k in ("lora_modules", "per_expert_rank", "init_scheme")}
                          for b in parents)
        else:
            key_sets = tuple(sorted(int(k) for k in b["A"]) for b in parents)
            layer_sets = key_sets

            def _rank(b, layers_b):
                a0 = b["A"][str(layers_b[0])]
                r = 1 if a0.dim() == 1 else int(a0.shape[1])
                if int(b.get("rank", r)) != r:
                    raise ValueError(
                        "lora_subtraction: a parent blob's rank key "
                        f"({b.get('rank')}) contradicts its tensors ({r})")
                return r
            ranks = tuple(_rank(b, ls) for b, ls in zip(parents, layer_sets))
            # Legacy parents carry no architecture keys beyond the optional
            # init scheme (FD 7): twins must share it.
            archs = tuple({"init_scheme": b.get("init_scheme", "zero")} for b in parents)
        fits = tuple((b.get("positions", "all"), b.get("response_header_len"))
                     for b in parents)

        def _key(k):
            return k if general[0] else str(k)
        cls._refuse_mismatch(
            layer_sets, ranks,
            tuple({(k, t): tuple(b[t][_key(k)].shape)
                   for k in ks for t in ("A", "B")}
                  for b, ks in zip(parents, key_sets)),
            fits, archs)
        keys = key_sets[0]
        blob = {"kind": cls.kind,
                "layers": layer_sets[0],
                "rank": ranks[0],
                **{f"{t}_{side}": {_key(k): b[t][_key(k)].clone() for k in keys}
                   for side, b in zip(("pos", "neg"), parents)
                   for t in ("A", "B")},
                "provenance": dict(provenance or {}),
                "positions": fits[0][0],
                "response_header_len": fits[0][1]}
        if general[0]:
            if pos_blob.get("targets") != neg_blob.get("targets"):
                raise ValueError(
                    "lora_subtraction: parents' `targets` blocks differ — not twins")
            for k in GENERAL_BLOB_KEYS:
                if k in pos_blob:
                    blob[k] = pos_blob[k]
        return blob

    @classmethod
    def load_blob(cls, blob: dict, model, tok) -> "LoRASubtractionObject":
        # Each parent re-hydrates through MultiLayerLoRAObject.load_blob —
        # the same validation (shape agreement, rank-key-vs-tensor
        # consistency, layer metadata) and the same op sequence at serve
        # time, so the positive side stays bitwise the mix fit.
        pos = MultiLayerLoRAObject.load_blob(cls._parent_blob(blob, "pos"),
                                             model, tok)
        neg = MultiLayerLoRAObject.load_blob(cls._parent_blob(blob, "neg"),
                                             model, tok)
        return cls(pos, neg, provenance=blob.get("provenance") or {})
