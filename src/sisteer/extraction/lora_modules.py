"""LoRA module sets: the vocabulary, per-model target resolution, the
per-expert rank rule and the trainable-parameter count.

The paper's LoRA adapts ONE module — the attention output projection
``o_proj`` of every full-attention block (:class:`MultiLayerLoRAObject`'s
historical shape). The LoRA architecture search (FD 6) widens that to a
MENU of module sets, each applied at EVERY eligible block (every block
that HAS the module):

=============  =====================================================  ======================================
set id         ``lora_modules`` (sorted short HF names)               what it reaches
=============  =====================================================  ======================================
``wo``         ``["o_proj"]``                                         attention output projection
``qkvo``       ``["k_proj", "o_proj", "q_proj", "v_proj"]``           the four attention projections
``attn_mlp``   attention four + ``down/gate/up_proj``                 attention + every FFN (shared + routed)
``mlp``        ``["down_proj", "gate_proj", "up_proj"]``              every FFN (shared + routed experts)
``down``       ``["down_proj"]``                                      every FFN's down projection
=============  =====================================================  ======================================

Resolution is a CAPABILITY PROBE over the live module tree, never a
model_type table (the ``models.oproj_layers`` precedent): a block
contributes a target iff it carries the named module. On the Qwen3.5
hybrids the Gated-DeltaNet blocks have no ``self_attn`` (their
``in_proj_*``/``out_proj`` are not Q/K/V/O and are never adapted); on
gemma-4 a KV-shared block may lack ``k_proj``/``v_proj`` — it then
contributes only the projections it has. The router ``gate``,
``shared_expert_gate`` and ``conv1d`` are never targets by construction
(nothing here looks for them).

MoE convention ("LoRA Without Regret", FD 6): the ROUTED experts of a
sparse-MoE FFN carry ONE 3-D LoRA per expert-parameter, uniform per-expert
rank ``r_e = max(1, r // active_experts)`` (with ``k`` active experts a rank
below ``k`` gives r_e = 1 and rank ``m·k`` gives ``m``: the tiny test model
with 2 active experts takes r_e = 2 at rank 4); the SHARED expert and every
attention/dense projection carry the full rank ``r``. HF's fused-experts
modules store the routed experts as two 3-D ``nn.Parameter``s
(``experts.gate_up_proj`` ``[E, 2I, d]`` and ``experts.down_proj``
``[E, d, I]``), so the trainable factors are the JOINT 3-D form the
serving engine consumes
(vLLM 0.25.1 ``FusedMoE3DWithLoRA``, ``vllm/lora/model_manager.py:835-880``;
``homework/05_vllm_lora_support.md`` §1d, §4): a shared ``A_gu[E, r_e, d]``
feeding the fused ``B_gu[E, 2I, r_e]`` (one rank-r_e bottleneck for gate
AND up together — the fused layout cannot express independent gate/up
factors) and ``A_d[E, r_e, I]`` / ``B_d[E, d, r_e]``. A set that names
``gate_proj``/``up_proj`` must name BOTH for a MoE block (refused
otherwise); the ``down`` set's routed target is down-only and its export
ships a ZERO gate_up companion because the engine asserts both expert
tensors at load (``model_manager.py:843``).

Parameter counts come from the built object's ACTUAL trainable tensors
(:func:`parameter_count`), never a hand-written table; the closed form
:func:`formula_parameter_count` exists so tests can pin the two against
each other on tiny dims and against hand counts on the verified
:data:`ARCH_DIMS_BY_MODEL_KEY` entries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from ..capeval.serving.serving_attestation import FFN_PARENTS  # the stdlib leaf: ONE FFN-parent vocabulary
from ..models import attention_oproj_site, get_decoder_layers
#: The module-set vocabulary (``MODULE_SETS``, set id -> sorted short HF names; ``ATTENTION_MODULES`` /
#: ``MLP_MODULES`` / ``KNOWN_MODULES``; :class:`LoRAModuleError`; :func:`normalise_modules`; :func:`set_id_for`) is DEFINED in
#: the stdlib leaf :mod:`.lora_module_sets` and re-exported here unchanged — one object per name in both modules — so a reader
#: that needs only the vocabulary (the steered-padding preflight's module-set selection, run in the suite's CPU/stdlib venv)
#: imports neither torch nor ``..models`` (transformers).
from .lora_module_sets import (ATTENTION_MODULES, KNOWN_MODULES,  # noqa: F401 — re-exported
                               MLP_MODULES, MODULE_SETS, LoRAModuleError,
                               normalise_modules, set_id_for)

#: The two routed-expert TARGET keys (the module half of an ``A``/``B`` blob
#: key ``"<layer>.<module>"``), spelled after the HF parameter names.
EXPERTS_GATE_UP = "experts.gate_up_proj"
EXPERTS_DOWN = "experts.down_proj"
EXPERT_TARGETS: tuple[str, ...] = (EXPERTS_GATE_UP, EXPERTS_DOWN)

#: The paper's LoRA architecture (FD 18): rank 4 on ``o_proj``.
PAPER_LORA_RANK = 4
PAPER_LORA_MODULES: tuple[str, ...] = MODULE_SETS["wo"]

INIT_SCHEMES: tuple[str, ...] = ("zero", "random_lecun_1e-3")
#: FD 7: the run override's LeCun-scaled random init — B ~ N(0, σ²) with
#: σ = LECUN_INIT_SCALE / sqrt(rank) (LoRA), v, w ~ N(0, σ²) with
#: σ = LECUN_INIT_SCALE / sqrt(d) (ReLU steer).
LECUN_INIT_SCALE = 1e-3


def lecun_sigma(n: int) -> float:
    """FD 7's σ = ``LECUN_INIT_SCALE / sqrt(n)``: ``n`` = the factor's rank
    for a LoRA B (the routed experts use THEIR factor's rank r_e — the
    LeCun logic keeps the delta's magnitude flat in the number of summed
    terms), the hidden size for the ReLU steer's v and w."""
    return LECUN_INIT_SCALE / float(int(n)) ** 0.5


def init_generator(*parts) -> torch.Generator:
    """One deterministic CPU RNG stream per ``parts`` — a tag and the
    (seed, layer[, module]) that identify an initial tensor — seeded from
    ``sha256("|".join(parts))``, so the same seed and architecture give the
    same draw in every process (the search trial, the production fit and
    the constitution/control twins of a seed; tests pin it). The LoRA
    object keys ``("lora_all", seed, layer, module)``, the ReLU steer
    ``("psr", seed, layer)``; the legacy o_proj object keeps its historical
    ``manual_seed(seed + layer)`` stream and never comes through here."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "big") % (1 << 62))


def per_expert_rank(rank: int, active_experts: int) -> int:
    """FD 6: ``max(1, rank // active_experts)`` — the uniform per-routed-
    expert rank; the shared expert keeps ``rank``."""
    rank = int(rank)
    active = int(active_experts)
    if rank < 1:
        raise LoRAModuleError(f"rank must be >= 1, got {rank}")
    if active < 1:
        raise LoRAModuleError(f"active_experts must be >= 1, got {active}")
    return max(1, rank // active)


_per_expert_rank_rule = per_expert_rank  # the kwarg below shadows the name


def per_expert_rank_map(rank: int, active_experts: int) -> dict[str, int]:
    """The ``per_expert_rank`` mapping the registry records:
    ``{"shared_expert": r, "routed_expert": max(1, r // active)}``."""
    return {"shared_expert": int(rank),
            "routed_expert": per_expert_rank(rank, active_experts)}


def routed_rank_of(per_expert_rank, rank: int) -> int | None:
    """The routed-expert rank a caller STATED, normalised: ``None`` (derive
    it from FD 6's rule), an int, or the registry's map ``{"shared_expert":
    r, "routed_expert": r_e}`` — whose ``shared_expert`` must equal
    ``rank`` (FD 6 keeps the full rank on the shared expert) and which must
    name ``routed_expert``. ONE normaliser for the object, the factory and
    the blob loader; the rule check against the model's ``active_experts``
    happens where the targets are known (``MultiLayerLoRAObject``)."""
    if per_expert_rank is None:
        return None
    if isinstance(per_expert_rank, Mapping):
        shared = per_expert_rank.get("shared_expert", rank)
        if int(shared) != int(rank):
            raise ValueError(
                f"per_expert_rank.shared_expert={shared} != rank={rank}: "
                "FD 6 keeps the FULL rank on the shared expert")
        if per_expert_rank.get("routed_expert") is None:
            raise ValueError("per_expert_rank map lacks 'routed_expert'")
        return int(per_expert_rank["routed_expert"])
    return int(per_expert_rank)


def _text_config(model_or_config):
    cfg = getattr(model_or_config, "config", model_or_config)
    text = getattr(cfg, "text_config", None)
    return text if text is not None else cfg


def active_experts_of(model_or_config) -> int | None:
    """``num_experts_per_tok`` of the (text) config, or None for a dense
    model — the divisor of the per-expert rank rule."""
    cfg = _text_config(model_or_config)
    for name in ("num_experts_per_tok", "top_k_experts"):
        v = getattr(cfg, name, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


# --------------------------------------------------------------------------
# Target resolution
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoRATarget:
    """One adapted (layer, module): ``module`` is the blob spelling
    (``o_proj`` … ``experts.down_proj``), ``site`` the dotted path of the
    module/parameter inside its decoder block (``self_attn.o_proj``,
    ``mlp.shared_expert.gate_proj``, ``mlp.gate_proj``,
    ``mlp.experts.gate_up_proj``), ``kind`` ``"linear"`` or ``"experts"``.
    Shapes are the base weight's: a linear has ``(in_features,
    out_features)``; an experts parameter additionally ``num_experts``.
    ``rank`` is the rank the factor is trained at (``r`` or ``r_e``)."""

    layer: int
    module: str
    site: str
    kind: str
    in_features: int
    out_features: int
    rank: int
    num_experts: int | None = None

    @property
    def key(self) -> str:
        """The ``A``/``B`` blob key: ``"<layer>.<module>"``."""
        return f"{self.layer}.{self.module}"

    @property
    def parameter_count(self) -> int:
        if self.kind == "experts":
            e = int(self.num_experts or 0)
            return e * self.rank * (self.in_features + self.out_features)
        return self.rank * (self.in_features + self.out_features)


def _linear_dims(mod: nn.Module) -> tuple[int, int] | None:
    w = getattr(mod, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        return None
    return int(w.shape[1]), int(w.shape[0])  # (in, out)


def _attention_parent(block: nn.Module) -> tuple[str, nn.Module] | None:
    site = attention_oproj_site(block)
    if site is None:
        return None
    parent_name = site.split(".")[0]
    return parent_name, getattr(block, parent_name)


def _ffn_layout(block: nn.Module) -> dict:
    """Where this block's FFN projections live: a dense ``mlp`` with
    gate/up/down, and/or a sparse block with ``shared_expert`` (+ 3-D
    ``experts`` parameters). Returns ``{"dense": module|None, "shared":
    module|None, "experts": module|None}`` with the attribute prefix each
    was found under."""
    out = {"dense": None, "shared": None, "experts": None,
           "dense_prefix": None, "shared_prefix": None, "experts_prefix": None}
    # the attestation's own tuple (serving_attestation.FFN_PARENTS): what this probe admits, the fingerprint's
    # MODULE_NAME_RE admits, so a set that fits here is never refused at readiness
    for ffn_name in FFN_PARENTS:
        ffn = getattr(block, ffn_name, None)
        if ffn is None or not isinstance(ffn, nn.Module):
            continue
        shared = getattr(ffn, "shared_expert", None)
        experts = getattr(ffn, "experts", None)
        if isinstance(shared, nn.Module) and any(
                isinstance(getattr(shared, m, None), nn.Module) for m in MLP_MODULES):
            out["shared"] = shared
            out["shared_prefix"] = f"{ffn_name}.shared_expert"
        if isinstance(experts, nn.Module) and isinstance(
                getattr(experts, "gate_up_proj", None), torch.Tensor) and isinstance(
                getattr(experts, "down_proj", None), torch.Tensor):
            gu, dn = experts.gate_up_proj, experts.down_proj
            if gu.dim() == 3 and dn.dim() == 3:
                out["experts"] = experts
                out["experts_prefix"] = f"{ffn_name}.experts"
        if out["shared"] is None and out["experts"] is None and any(
                isinstance(getattr(ffn, m, None), nn.Module) for m in MLP_MODULES):
            # A dense FFN (Llama/gemma ``mlp``, NemotronH's MLP ``mixer``):
            # only when it is not the MoE wrapper itself.
            out["dense"] = ffn
            out["dense_prefix"] = ffn_name
        if out["dense"] or out["shared"] or out["experts"]:
            break
    return out


def resolve_targets(model: nn.Module, modules: Iterable[str] | str, *,
                    rank: int, layers: Sequence[int] | None = None,
                    per_expert_rank: int | None = None,
                    ) -> list[LoRATarget]:
    """Every (layer, module) the set reaches on ``model`` — the capability
    probe of the module docstring. ``layers`` restricts the blocks (an
    explicit member that carries NONE of the set's modules is a caller
    error and refuses); ``per_expert_rank`` defaults to the FD 6 rule from
    the config's ``num_experts_per_tok`` and only matters when routed
    experts are reached. Refuses a set that resolves to nothing."""
    mods = normalise_modules(modules)
    rank = int(rank)
    if rank < 1:
        raise LoRAModuleError(f"rank must be >= 1, got {rank}")
    blocks = get_decoder_layers(model)
    wanted = list(range(len(blocks))) if layers is None else [int(l) for l in layers]
    for l in wanted:
        if l < 0 or l >= len(blocks):
            raise LoRAModuleError(
                f"layer {l} is outside the model's {len(blocks)} decoder blocks")
    attn_wanted = [m for m in ATTENTION_MODULES if m in mods]
    mlp_wanted = [m for m in MLP_MODULES if m in mods]
    gate_up_wanted = "gate_proj" in mods or "up_proj" in mods
    if gate_up_wanted and not ("gate_proj" in mods and "up_proj" in mods):
        # Only matters on MoE blocks (the fused factor), but a set with
        # exactly one of the pair is not one of the five and is refused
        # up front so the search vocabulary stays closed.
        raise LoRAModuleError(
            f"module set {list(mods)} names one of gate_proj/up_proj without "
            "the other — the routed experts' fused gate_up factor cannot "
            "express a gate-only or up-only adapter")
    r_e = per_expert_rank
    targets: list[LoRATarget] = []
    for l in wanted:
        block = blocks[l]
        found_here = False
        if attn_wanted:
            parent = _attention_parent(block)
            if parent is not None:
                pname, pmod = parent
                for m in attn_wanted:
                    sub = getattr(pmod, m, None)
                    dims = _linear_dims(sub) if isinstance(sub, nn.Module) else None
                    if dims is None:
                        continue  # a KV-shared gemma-4 block without k/v_proj
                    targets.append(LoRATarget(l, m, f"{pname}.{m}", "linear",
                                              dims[0], dims[1], rank))
                    found_here = True
        if mlp_wanted:
            ffn = _ffn_layout(block)
            for which in ("dense", "shared"):
                host = ffn[which]
                if host is None:
                    continue
                prefix = ffn[f"{which}_prefix"]
                for m in mlp_wanted:
                    sub = getattr(host, m, None)
                    dims = _linear_dims(sub) if isinstance(sub, nn.Module) else None
                    if dims is None:
                        continue
                    targets.append(LoRATarget(l, m, f"{prefix}.{m}", "linear",
                                              dims[0], dims[1], rank))
                    found_here = True
            experts = ffn["experts"]
            if experts is not None:
                if r_e is None:
                    active = active_experts_of(model)
                    if active is None:
                        raise LoRAModuleError(
                            f"layer {l} carries routed experts but the config "
                            "has no num_experts_per_tok — pass per_expert_rank")
                    r_e = _per_expert_rank_rule(rank, active)
                r_e = int(r_e)
                if r_e < 1:
                    raise LoRAModuleError(f"per_expert_rank must be >= 1, got {r_e}")
                gu, dn = experts.gate_up_proj, experts.down_proj
                E, two_i, d = (int(x) for x in gu.shape)
                E2, d2, inter = (int(x) for x in dn.shape)
                if E != E2 or d != d2 or two_i != 2 * inter:
                    raise LoRAModuleError(
                        f"layer {l}: experts.gate_up_proj {tuple(gu.shape)} and "
                        f"experts.down_proj {tuple(dn.shape)} are not the "
                        "[E, 2I, d] / [E, d, I] pair the fused layout assumes")
                prefix = ffn["experts_prefix"]
                if gate_up_wanted:
                    targets.append(LoRATarget(l, EXPERTS_GATE_UP, f"{prefix}.gate_up_proj",
                                              "experts", d, two_i, r_e, E))
                    found_here = True
                if "down_proj" in mods:
                    targets.append(LoRATarget(l, EXPERTS_DOWN, f"{prefix}.down_proj",
                                              "experts", inter, d, r_e, E))
                    found_here = True
        if layers is not None and not found_here:
            raise LoRAModuleError(
                f"layer {l} carries none of {list(mods)} (a linear-attention, "
                "Mamba or MLP-less block?) — the explicit layer list names a "
                "block this set cannot adapt; refusing to narrow it silently")
    if not targets:
        raise LoRAModuleError(
            f"module set {list(mods)} resolves to no module on "
            f"{type(model).__name__} — refusing to construct a silent no-op object")
    return targets


def targets_summary(targets: Sequence[LoRATarget]) -> dict[str, dict]:
    """The ``targets`` block of a lora_all blob: ``{module: {site, layers,
    rank, kind[, num_experts]}}`` in module order of first appearance."""
    out: dict[str, dict] = {}
    for t in targets:
        rec = out.setdefault(t.module, {"site": t.site, "layers": [], "rank": int(t.rank),
                                        "kind": t.kind})
        if t.kind == "experts":
            rec.setdefault("num_experts", int(t.num_experts or 0))
        if rec["site"] != t.site or rec["rank"] != int(t.rank):
            raise LoRAModuleError(
                f"module {t.module!r} resolves to different sites/ranks across "
                f"layers ({rec['site']}@{rec['rank']} vs {t.site}@{t.rank})")
        rec["layers"].append(int(t.layer))
    for rec in out.values():
        rec["layers"] = sorted(rec["layers"])
    return out


def layers_by_module(targets: Sequence[LoRATarget]) -> dict[str, list[int]]:
    """``fit_layers_json``'s shape: adapted block indices per module."""
    return {m: rec["layers"] for m, rec in targets_summary(targets).items()}


# --------------------------------------------------------------------------
# Parameter counts
# --------------------------------------------------------------------------

def parameter_count(obj) -> int:
    """The number of TRAINABLE scalars of a built object (``obj.trainable``)
    — the ``trainable_params`` the table and the registry carry. Zero
    companions (the ``down`` set's frozen gate_up pair) are not trainable
    and are therefore not counted."""
    return int(sum(int(p.numel()) for p in obj.trainable))


@dataclass(frozen=True)
class ArchDims:
    """The widths the closed-form count needs. ``attention_layers`` /
    ``moe_layers`` / ``dense_mlp_layers`` are COUNTS of blocks; the
    attention widths are the projections' output dims (``q_out`` carries
    Qwen3.5's doubled query + output-gate width)."""

    hidden: int
    attention_layers: int
    q_out: int
    kv_out: int
    o_in: int
    moe_layers: int = 0
    num_experts: int = 0
    active_experts: int = 1
    moe_intermediate: int = 0
    shared_intermediate: int = 0
    dense_mlp_layers: int = 0
    dense_intermediate: int = 0


#: The verified dims BY MODEL KEY — the one table the serving probe plans
#: from (scripts/sdfbase/serving_probe.py ``dims_for``; its probe_plan.json
#: records ``dataclasses.asdict`` of the entry) and the padded preflight sizes
#: its warmup MLP-intermediate term from (the HF config is read only BEHIND
#: it, by architecture). An onboarded model is added here; a key with no
#: entry is None for readers that can fall back to the config and a refusal
#: by name for those that cannot.
#: Qwen3.5-4B (reproduce/models.yaml `architecture`, verified on the pinned
#: checkpoint when the row was written: 32 blocks, hidden 2560, 8 full-attention
#: blocks carrying self_attn.o_proj (the other 24 are gated-deltanet linear
#: attention), gated attention q_out 2 x 16 x 256, 4 KV heads x 256, o_in
#: 16 x 256, dense MLP intermediate 9,216 on every block). The worked
#: example's served key `qwen35_4b_demo` serves this checkpoint.
QWEN35_4B_DIMS = ArchDims(hidden=2560, attention_layers=8, q_out=2 * 16 * 256,
                          kv_out=4 * 256, o_in=16 * 256,
                          dense_mlp_layers=32, dense_intermediate=9_216)

ARCH_DIMS_BY_MODEL_KEY: Mapping[str, ArchDims] = {
    "qwen35_4b_demo": QWEN35_4B_DIMS,
}


def arch_dims_for(model_key: str) -> ArchDims | None:
    """The verified :class:`ArchDims` of ``model_key`` (None: no entry)."""
    return ARCH_DIMS_BY_MODEL_KEY.get(model_key)


def formula_parameter_count(dims: ArchDims, modules: Iterable[str] | str,
                            rank: int) -> int:
    """Closed-form trainable count for a module set at ``rank`` on
    ``dims``: attention per block
    ``r·(q: d + q_out) + r·(k: d + kv_out) + r·(v: d + kv_out) + r·(o: o_in + d)``
    over the adapted projections; dense/shared FFN ``r·(d + I)`` per
    adapted projection; routed experts ``E·r_e·(d + 2I)`` for gate_up and
    ``E·r_e·(I + d)`` for down per MoE block. Tests pin it against the
    tensors of a built object on tiny dims and against hand counts on the
    real ones."""
    mods = normalise_modules(modules)
    r = int(rank)
    d = dims.hidden
    total = 0
    attn = 0
    if "q_proj" in mods:
        attn += r * (d + dims.q_out)
    if "k_proj" in mods:
        attn += r * (d + dims.kv_out)
    if "v_proj" in mods:
        attn += r * (d + dims.kv_out)
    if "o_proj" in mods:
        attn += r * (dims.o_in + d)
    total += attn * dims.attention_layers
    mlp = [m for m in MLP_MODULES if m in mods]
    if mlp:
        if dims.dense_mlp_layers:
            total += dims.dense_mlp_layers * len(mlp) * r * (d + dims.dense_intermediate)
        if dims.moe_layers:
            shared = len(mlp) * r * (d + dims.shared_intermediate)
            r_e = per_expert_rank(r, dims.active_experts)
            routed = 0
            if "gate_proj" in mods:  # gate+up together (normalise refuses one alone)
                routed += dims.num_experts * r_e * (d + 2 * dims.moe_intermediate)
            if "down_proj" in mods:
                routed += dims.num_experts * r_e * (dims.moe_intermediate + d)
            total += dims.moe_layers * (shared + routed)
    return int(total)


def set_order_by_parameter_count(counts: Mapping[str, int]) -> list[str]:
    """The ORDINAL order of module sets the selection guard uses for its
    narrowest/widest alarm — derived from trainable-parameter counts at
    run time (``{set_id: trainable_params}``), never a hand-written list;
    ties broken by name so the order is total."""
    return sorted(counts, key=lambda s: (int(counts[s]), s))
