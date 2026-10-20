"""Per-server ADAPTER MANIFESTS for the native multi-LoRA standups, the
expected-bytes law of their LoRA buffers, and the packed-stride /
KV-derived admission of ``max_num_seqs``.

Every native standup serves ONE engine whose adapters all name ONE
``--lora-target-modules`` class (``vllm_server._adapted_module_class``) and
whose LoRA buffers are pre-allocated at engine init from the ONE global
``max_lora_rank`` cap for every slot of ``max_loras``. A manifest is the
explicit list of that engine's adapters (the genuine LoRA cells, the
mandatory zero adapter, the residual-slot carriers), its class, its slot
count and its cap — decided BEFORE any GPU minute, so the serving probe
stands up the complete mixed manifests the production waves will run,
not isolated synthetic adapters.

Topology per module set (the rule of 2026-10-01 00:22Z —
the difference adapter is served on ITS OWN server at cap 2r, the control and
constitution adapters on a server at cap r, for EVERY module set, so a
non-difference adapter's dense ``[R, in]`` / ``[out, R]`` slices are never
padded to 2r; the probe's former S1a/S1b/S1c fallback shape, promoted to the
primary and merged two-into-one):

``wo``
    TWO standups per view. ``S1cc``: control r + constitution r + the zero
    adapter + 4 residual carriers (refusal direction, 3 ReLU steers as rank-1
    ``o_proj`` zeros) = 7 slots, class ``o_proj``, cap = the smallest legal
    value >= r. ``S1d``: the difference (2r) + the zero adapter = 2 slots,
    class ``o_proj``, cap >= 2r.
``qkvo`` / ``attn_mlp`` / ``mlp`` / ``down``
    THREE standups per view. ``S1cc`` (lora, single-rank): control +
    constitution + zero = 3 slots, class = the set's modules (plus
    ``experts`` when routed experts are adapted), cap >= r. ``S1d`` (lora,
    difference): difference (2r) + zero = 2 slots, same class, cap >= 2r.
    ``S2`` (residual): ``none`` (the zero adapter, on which the two SI
    passthroughs ride) + the refusal direction + 3 ReLU carriers as rank-1
    ``o_proj`` zeros = 5 registered slots, class ``o_proj``, cap 8.

LEGACY labels a RECORD may still carry (``Manifest.from_dict`` reads them; the
verdict certifies a pair ROLE by ROLE over whatever manifests its record
planned): ``single`` (the old one-server ``wo``: cells + zero + carriers at
cap 2r), ``S1`` (the old four-slot server: control + constitution +
difference + zero at cap 2r) and its three-server VRAM fallback ``S1a`` /
``S1b`` / ``S1c``. ``build_manifests`` no longer emits them.

THE SERVED LAUNCH CLASS (the probe owner's ruling of 2026-10-11): a steered
native standup registers EVERY exported block of its view plus ``none__lora0``
in ONE engine — for ``wo`` the three cells, the zero adapter and the four
carriers (8 slots at cap >= 2r), for the other sets the three cells and the zero
adapter (4 slots; the carriers ride S2) — and the site record sizes it from the
probed launches whose engine class DOMINATES that one at the SAME window (rank
cap >= and slots >=; ``geometry.least_admitted_by_window``, no window transfer).
The split servers never do: ``S1cc`` has 7 (or 3) slots, ``S1d`` 2. So beside
them the probe plans the pair's UNION manifest (:func:`build_union_manifest`):
the ``single`` shape above, one natural launch at EVERY plan window, folded
beside the servers; it certifies the CLASS, while the roles stay certified by
``S1cc`` / ``S1d`` (the verdict orders ``single`` after them).

THE ROUTED-EXPERT BUFFER RANK. When the class wraps the routed experts
every manifest carries ``moe_lora_rank``: the rank the server's fused-MoE LoRA
buffers are allocated at — the LARGEST per-expert rank over its adapters (the
difference's ``2·r_e``, the zero adapter's ``cap // active``), never the dense
cap. The composer derives the same number from the adapters' own
``rank_pattern`` (``vllm_server.fitted_moe_lora_rank``) and the worker
allocates at it (``native_moe_lora``); the bytes law below prices the expert
family at it. At r = 64 on an 8-active model: S1cc caps at 64 with experts at
R = 8, S1d at 128 with experts at R = 16 — against a four-slot S1 whose experts
were padded to R = 128, sixteen times their need. None of the paper models
routes experts; the terms stay as an OPTION for a checkpoint that does.

The bytes law (:func:`expected_lora_bytes`) is derived per manifest from
the wrapped classes' STACKED buffer shapes at vLLM 0.25.1, bf16:

* routed experts (``FusedMoE3DWithLoRA``, ``vllm/lora/layers/fused_moe.py``
  ``_create_lora_a_weights`` :157-189 / ``FusedMoE3DWithLoRA
  ._create_lora_b_weights`` :466-490; ``_w13_slices = 1`` :464): per MoE
  layer and slot ``w13_lora_a [E, R, d]`` (ONE shared A slice for gate+up),
  ``w13_lora_b [E, 2I, R]``, ``w2_lora_a [E, R, I]``, ``w2_lora_b [E, d, R]``
  -> ``E·R·(d + 2I + I + d)·2 B``, at the manifest's ``moe_lora_rank``;
* a packed linear (``MergedQKVParallelLinearWithLoRA`` /
  ``MergedColumnParallelLinearWithLoRA``, ``column_parallel_linear.py``
  ``create_lora_weights`` :225-245): ``lora_a_stacked`` is a TUPLE with ONE
  ``[R, in]`` tensor PER SLICE and ``lora_b_stacked`` one ``[out_i, R]`` per
  slice -> qkv on the worked example's Qwen3.5-4B dims ``3·R·2560 + R·(8192 +
  2·1024)`` per attention block, the dense MLP's packed gate_up ``2·R·2560 +
  2·9216·R`` per block (a MoE block's shared expert the same with its own
  intermediate); plain projections (``RowParallelLinearWithLoRA`` o_proj /
  down_proj) ``R·in + out·R``; so ``wo`` ≈ 0.10 MiB × R and ``qkvo`` ≈ 0.375
  MiB × R per slot over the 8 full-attention blocks;
* everything multiplied by ``max_loras`` (buffers are per slot, allocated
  whenever the class wraps the module, regardless of what each adapter
  actually adapts); the dense families at the cap ``R``, the routed-expert
  family at the manifest's ``moe_lora_rank``.

The admission of ``max_num_seqs`` per eval window (:func:`admitted_max_num_seqs`)
is the harness's int32 kernel index ceiling on the PACKED EXECUTING stride
(``serving_attestation.LORA_KERNEL_INDEX_LIMIT // D``: the qkv wrapper's
``sum(output_slices)`` = 8192 + 2 × 1024 when Q/K/V are adapted on the
Qwen3.5-4B dims -> S <= 25 at L = 8,192 and S <= 4 at 49,152; ``wo`` keeps
o_proj's 4,096-in stride -> S <= 63 / S <= 10) together with the KV budget
(32 KiB/token on those dims: 8 full-attention blocks × 4 KV heads × 256 × K+V ×
2 B; 1.5 GiB per 49,152-token sequence) plus, for a hybrid with a recurrent
state, a constant per-sequence term the caller derives from the checkpoint's
config (:func:`gated_deltanet_state_per_seq_bytes`) — never a typed per-model
constant.

Pure Python over ``lora_modules.ArchDims``-shaped inputs (any object with
``hidden, attention_layers, q_out, kv_out, o_in, moe_layers, num_experts,
active_experts, moe_intermediate, shared_intermediate, dense_mlp_layers,
dense_intermediate``); no vllm, no peft, no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sisteer.extraction.lora_modules import (ATTENTION_MODULES, MLP_MODULES, MODULE_SETS,
                                             formula_parameter_count, normalise_modules,
                                             per_expert_rank, per_expert_rank_map,
                                             set_id_for)

#: the harness's int32 LoRA-kernel index limit and the vLLM wrapper-target word
#: for the routed experts (``FusedMoE`` suffix; NEVER a PEFT ``target_modules``
#: entry — homework 05 §1d; contract §3): the attestation vocabulary's, not a
#: second spelling
from .serving_attestation import (EXPERTS_LEAF as EXPERTS_TARGET, LORA_KERNEL_INDEX_LIMIT,
                                  lora_kernel_index_ceiling)
#: vLLM 0.25.1 ``MaxLoRARanks`` (``vllm/config/lora.py:26``) — the only legal
#: ``--max-lora-rank`` values: the ONE tuple the composer fits the cap from
from .vllm_server import LEGAL_LORA_RANK_CAPS
#: the cap ladder the probe walks as SEPARATE engine launches
RANK_CAP_LADDER: Tuple[int, ...] = (8, 16, 32, 64, 128)
#: the production eval windows: statics, PAIR, Crescendo / ActorAttack, the
#: 65,536 window and the padding ladder's 98,304 ceiling
PRODUCTION_WINDOWS: Tuple[int, ...] = (4096, 16384, 49152, 65536, 98304)

BF16_BYTES = 2
#: nvidia-smi ``memory.total`` of an H200 SXM in MiB (141 GB HBM3e)
H200_TOTAL_BYTES = 143_771 * 2**20
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
#: a model-neutral CONSERVATIVE default for the served model's bf16 base
#: weights (a ~30B-class model with headroom). The probe's ``--base-weights-gb``
#: overrides it and the plan's budget block records the value every launch was
#: planned with.
DEFAULT_BASE_WEIGHTS_BYTES = 70e9
#: CUDA graphs + activations headroom the plan budgets
DEFAULT_GRAPHS_ACTIVATIONS_BYTES = 6e9
#: the HF text-config fields of a Gated-DeltaNet hybrid (Qwen3.5's ``linear_attention``
#: blocks) that :func:`gated_deltanet_state_per_seq_bytes` reads
GDN_CONFIG_FIELDS: Tuple[str, ...] = ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                                      "linear_value_head_dim", "linear_conv_kernel_dim")

#: LEGACY labels (records written before 2026-10-01): the one-server ``wo``, the
#: four-slot LoRA server and its three-server VRAM fallback
TOPOLOGY_SINGLE = "single"
TOPOLOGY_S1 = "S1"
TOPOLOGY_S2 = "S2"
#: the DEFAULT topology (the 2026-10-01 ruling): the single-rank server (control +
#: constitution + zero [+ the carriers for ``wo``] at cap r) and the difference
#: server (difference + zero at cap 2r)
TOPOLOGY_S1CC = "S1cc"
TOPOLOGY_S1D = "S1d"
#: the three-server S1 fallback (LEGACY): ONE genuine cell per server, keyed by the
#: cell's ROLE (the verdict finds the server that certified a role through
#: this map, never through tuple position)
S1_FALLBACK_BY_ROLE: Dict[str, str] = {"control": "S1a", "constitution": "S1b",
                                       "difference": "S1c"}
S1_FALLBACKS: Tuple[str, ...] = tuple(S1_FALLBACK_BY_ROLE.values())
TOPOLOGY_LABEL_TWO_SERVER = "S1S2"
#: the canonical order of topology labels in a pair's ``topology`` string
TOPOLOGY_ORDER: Tuple[str, ...] = (TOPOLOGY_SINGLE, TOPOLOGY_S1, "S1a", "S1b", "S1c", TOPOLOGY_S1CC, TOPOLOGY_S1D,
                                   TOPOLOGY_S2)
#: the genuine roles of a (set, rank) pair; ``ROLES_REQUIRED`` are the two the verdict
#: certifies by name (the single-rank adapter and the rank-2r difference — as before)
ROLE_CONTROL, ROLE_CONSTITUTION, ROLE_DIFFERENCE = "control", "constitution", "difference"
GENUINE_ROLES: Tuple[str, ...] = (ROLE_CONTROL, ROLE_CONSTITUTION, ROLE_DIFFERENCE)
ROLES_REQUIRED: Tuple[str, ...] = (ROLE_CONTROL, ROLE_DIFFERENCE)
#: S2's cap: "max_lora_rank 8" for the rank-1 o_proj carriers
S2_RANK_CAP = 8

KIND_GENUINE = "genuine"
KIND_ZERO = "zero"
KIND_CARRIER = "carrier"
ADAPTER_KINDS: Tuple[str, ...] = (KIND_GENUINE, KIND_ZERO, KIND_CARRIER)

#: the S2 cells that ride the zero adapter (prompted passthroughs get no slot
#: of their own in the native standup: ``none__lora0`` serves them)
S2_PASSTHROUGH_CELLS: Tuple[str, ...] = ("si_safety", "si_constitution")
#: the residual-slot carriers of a view: the refusal direction + 3 ReLU steers
CARRIER_ROLES: Tuple[str, ...] = ("refusal_direction", "relu_control",
                                  "relu_constitution", "relu_difference")


class ManifestError(ValueError):
    """A manifest the topology cannot express (unknown set, rank outside the
    legal caps, an S1 fallback asked for a set that has none)."""


def rank_cap(max_rank: int) -> int:
    """The smallest legal ``--max-lora-rank`` >= ``max_rank``
    (``vllm_server.fitted_max_lora_rank``'s rule over the same table)."""
    r = int(max_rank)
    if r < 1:
        raise ManifestError(f"rank must be >= 1, got {r}")
    for cap in LEGAL_LORA_RANK_CAPS:
        if cap >= r:
            return cap
    raise ManifestError(f"rank {r} exceeds the largest legal --max-lora-rank "
                        f"{LEGAL_LORA_RANK_CAPS[-1]} (vllm/config/lora.py MaxLoRARanks @0.25.1)")


def _mlp_modules(modules: Sequence[str]) -> List[str]:
    return [m for m in MLP_MODULES if m in modules]


def _attention_modules(modules: Sequence[str]) -> List[str]:
    return [m for m in ATTENTION_MODULES if m in modules]


def adapts_routed_experts(dims: Any, modules: Iterable[str]) -> bool:
    """True iff the set reaches a routed-experts FFN on ``dims`` (a MoE
    model and any MLP projection in the set)."""
    mods = normalise_modules(modules)
    return bool(getattr(dims, "moe_layers", 0)) and bool(_mlp_modules(mods))


def target_class_for(dims: Any, modules: Iterable[str]) -> Tuple[str, ...]:
    """The vLLM ``--lora-target-modules`` class of a set on ``dims``: the
    set's PEFT short names plus ``experts`` when routed experts are adapted
    (the harness infers it from the attested expert factor keys; an
    omission loads and silently no-ops — homework 05 §1d); a dense model's
    class is the set's names, sorted."""
    mods = list(normalise_modules(modules))
    if adapts_routed_experts(dims, mods):
        mods.append(EXPERTS_TARGET)
    return tuple(sorted(mods))


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AdapterSlot:
    """One registered adapter of a standup. ``lora_modules`` are the PEFT
    short names (never ``experts``); ``rank`` is the adapter's top-level
    ``r`` (2r for a difference; the cap for the zero adapter, so the fitted
    cap equals the planned one); ``per_expert_rank`` the routed r_e when
    experts are adapted (None otherwise — every slot of a dense model)."""

    name: str
    kind: str
    lora_modules: Tuple[str, ...]
    rank: int
    role: str
    per_expert_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in ADAPTER_KINDS:
            raise ManifestError(f"adapter {self.name!r}: kind {self.kind!r} not in {ADAPTER_KINDS}")
        if int(self.rank) < 1:
            raise ManifestError(f"adapter {self.name!r}: rank {self.rank} < 1")

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "lora_modules": list(self.lora_modules),
                "rank": int(self.rank), "per_expert_rank": self.per_expert_rank, "role": self.role}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AdapterSlot":
        return cls(name=str(d["name"]), kind=str(d["kind"]),
                   lora_modules=tuple(str(m) for m in d["lora_modules"]), rank=int(d["rank"]),
                   per_expert_rank=(None if d.get("per_expert_rank") is None
                                    else int(d["per_expert_rank"])),
                   role=str(d["role"]))


@dataclass(frozen=True)
class Manifest:
    """One standup's complete adapter list + the engine facts they imply."""

    name: str
    set_id: str
    rank: int
    topology: str
    adapters: Tuple[AdapterSlot, ...]
    target_class: Tuple[str, ...]
    max_loras: int
    max_lora_rank: int
    fallback_of: Optional[str] = None
    passthrough_cells: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    #: THE ROUTED-EXPERT BUFFER RANK: the R the server's fused-MoE LoRA
    #: buffers are allocated at when the class wraps the routed experts — the
    #: largest ``per_expert_rank`` over the adapters (derived when None is given);
    #: None iff the class wraps no expert. An adapter whose routed rank exceeds
    #: it is refused by name here, exactly as the worker refuses it at load.
    moe_lora_rank: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_lora_rank not in LEGAL_LORA_RANK_CAPS:
            raise ManifestError(f"{self.name}: cap {self.max_lora_rank} is not a legal --max-lora-rank")
        routed = [int(a.per_expert_rank) for a in self.adapters if a.per_expert_rank is not None]
        if self.wraps_experts:
            need = max(routed) if routed else None
            if need is None:
                raise ManifestError(f"{self.name}: the class wraps the routed experts but no adapter states a per_expert_rank")
            if self.moe_lora_rank is None:
                object.__setattr__(self, "moe_lora_rank", int(need))
            elif int(self.moe_lora_rank) < need:
                raise ManifestError(f"{self.name}: moe_lora_rank {self.moe_lora_rank} < an adapter's routed rank {need} — "
                                    "a wider adapter is never truncated into the expert buffers")
            if int(self.moe_lora_rank) > int(self.max_lora_rank):
                raise ManifestError(f"{self.name}: moe_lora_rank {self.moe_lora_rank} > cap {self.max_lora_rank}")
        elif self.moe_lora_rank is not None:
            raise ManifestError(f"{self.name}: moe_lora_rank {self.moe_lora_rank} stated but the class wraps no expert")
        if self.max_loras != len(self.adapters):
            raise ManifestError(f"{self.name}: max_loras {self.max_loras} != {len(self.adapters)} adapters "
                                "(the harness composes --max-loras from the registered count)")
        names = [a.name for a in self.adapters]
        if len(set(names)) != len(names):
            raise ManifestError(f"{self.name}: duplicate adapter names {names}")
        largest = max(a.rank for a in self.adapters)
        if largest > self.max_lora_rank:
            raise ManifestError(f"{self.name}: adapter rank {largest} > cap {self.max_lora_rank}")
        if not any(a.kind == KIND_ZERO for a in self.adapters):
            raise ManifestError(f"{self.name}: every native standup carries the zero adapter")

    @property
    def genuine(self) -> Tuple[AdapterSlot, ...]:
        return tuple(a for a in self.adapters if a.kind == KIND_GENUINE)

    @property
    def zero(self) -> AdapterSlot:
        return next(a for a in self.adapters if a.kind == KIND_ZERO)

    @property
    def wraps_experts(self) -> bool:
        return EXPERTS_TARGET in self.target_class

    @property
    def launch_key(self) -> str:
        """The ENGINE configuration this manifest composes — identical for
        two manifests whose engines would be byte-identical (every S2 of
        every MLP-bearing set and rank), so the probe launches it once. A
        dense manifest's key is spelled exactly as before the expert terms
        existed (no routed rank, ``0`` in the slot's r_e position)."""
        slots = sorted((a.kind, a.rank, a.per_expert_rank or 0, ",".join(a.lora_modules))
                       for a in self.adapters)
        return (f"{self.topology}|{','.join(self.target_class)}|cap{self.max_lora_rank}|"
                + (f"moe{self.moe_lora_rank}|" if self.moe_lora_rank is not None else "")
                + ";".join(f"{k}:{r}:{re}:{m}" for k, r, re, m in slots))

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "set": self.set_id, "rank": int(self.rank), "topology": self.topology,
                "adapters": [a.as_dict() for a in self.adapters],
                "target_class": list(self.target_class), "max_loras": int(self.max_loras),
                "max_lora_rank": int(self.max_lora_rank), "fallback_of": self.fallback_of,
                "passthrough_cells": list(self.passthrough_cells), "notes": list(self.notes),
                "moe_lora_rank": (int(self.moe_lora_rank) if self.moe_lora_rank is not None else None),
                "launch_key": self.launch_key}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Manifest":
        return cls(name=str(d["name"]), set_id=str(d["set"]), rank=int(d["rank"]),
                   topology=str(d["topology"]),
                   adapters=tuple(AdapterSlot.from_dict(a) for a in d["adapters"]),
                   target_class=tuple(str(t) for t in d["target_class"]),
                   max_loras=int(d["max_loras"]), max_lora_rank=int(d["max_lora_rank"]),
                   fallback_of=d.get("fallback_of"),
                   passthrough_cells=tuple(d.get("passthrough_cells") or ()),
                   notes=tuple(d.get("notes") or ()),
                   # an older record carries no moe_lora_rank: the adapters' need is derived (its engine
                   # ran at the cap then; the launch records it certified keep their recorded expected bytes)
                   moe_lora_rank=(None if d.get("moe_lora_rank") is None else int(d["moe_lora_rank"])))

    @property
    def genuine_roles(self) -> Tuple[str, ...]:
        return tuple(a.role for a in self.adapters if a.kind == KIND_GENUINE)

    def carries_role(self, role: str) -> bool:
        """True iff a GENUINE adapter of ``role`` is served by this manifest."""
        return role in self.genuine_roles


def _r_e(dims: Any, modules: Sequence[str], rank: int) -> Optional[int]:
    if not adapts_routed_experts(dims, modules):
        return None
    return per_expert_rank(rank, int(getattr(dims, "active_experts", 1) or 1))


def _genuine(name: str, role: str, modules: Tuple[str, ...], rank: int, dims: Any,
             per_expert_rank: Optional[int] = None) -> AdapterSlot:
    return AdapterSlot(name=name, kind=KIND_GENUINE, lora_modules=modules, rank=int(rank),
                       per_expert_rank=(_r_e(dims, modules, rank) if per_expert_rank is None else int(per_expert_rank)),
                       role=role)


def difference_per_expert_rank(dims: Any, modules: Sequence[str], rank: int) -> Optional[int]:
    """The DIFFERENCE adapter's routed rank: ``2 · r_e(r)`` — the two rank-``r_e``
    parents stacked along the per-expert rank axis (``lora_export.
    _export_general_subtraction``: ``rank_pattern`` is ``2·r_e``), NOT
    ``r_e(2r)``; the two agree for r >= active_experts and differ at r = 4
    (2 vs 1 with 8 active experts). None when no routed expert is reached."""
    r_e = _r_e(dims, modules, rank)
    return None if r_e is None else 2 * int(r_e)


def _zero(modules: Tuple[str, ...], cap: int, dims: Any) -> AdapterSlot:
    # rank = the cap: the harness fits --max-lora-rank to the LARGEST declared r
    # (vllm_server.fitted_max_lora_rank), so the zero adapter pins the planned cap
    return AdapterSlot(name="none__lora0", kind=KIND_ZERO, lora_modules=modules, rank=int(cap),
                       per_expert_rank=_r_e(dims, modules, cap), role="none")


def _carriers() -> Tuple[AdapterSlot, ...]:
    return tuple(AdapterSlot(name=f"{role}__carrier", kind=KIND_CARRIER, lora_modules=("o_proj",),
                             rank=1, per_expert_rank=None, role=role) for role in CARRIER_ROLES)


def _cells(modules: Tuple[str, ...], rank: int, dims: Any) -> Tuple[AdapterSlot, ...]:
    return (_genuine("control__lora", ROLE_CONTROL, modules, rank, dims),
            _genuine("constitution__lora", ROLE_CONSTITUTION, modules, rank, dims),
            _genuine("difference__lora", ROLE_DIFFERENCE, modules, 2 * rank, dims,
                     per_expert_rank=difference_per_expert_rank(dims, modules, rank)))


def build_s2_manifest(set_id: str, rank: int) -> Manifest:
    """The residual server every MLP-bearing (and ``qkvo``) set shares: the
    zero adapter (``none``; the SI passthroughs ride it) + the four rank-1
    ``o_proj`` carriers, class ``o_proj``, cap 8 (``max_lora_rank 8``; the
    composed ``--max-loras`` is the registered count, 5, not the nominal
    8)."""
    adapters = (_zero(("o_proj",), S2_RANK_CAP, None),) + _carriers()
    return Manifest(name=f"{set_id}_r{rank}_{TOPOLOGY_S2}", set_id=set_id, rank=int(rank),
                    topology=TOPOLOGY_S2, adapters=adapters, target_class=("o_proj",),
                    max_loras=len(adapters), max_lora_rank=S2_RANK_CAP,
                    passthrough_cells=S2_PASSTHROUGH_CELLS,
                    notes=("no MoE buffers: class o_proj only",
                           "the two SI passthroughs ride the zero adapter (no slot of their own)"))


def build_manifests(dims: Any, *, set_id: str, rank: int,
                    kv_budget_bytes: Optional[float] = None,
                    base_weights_bytes: float = DEFAULT_BASE_WEIGHTS_BYTES,
                    graphs_activations_bytes: float = DEFAULT_GRAPHS_ACTIVATIONS_BYTES,
                    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
                    total_vram_bytes: float = H200_TOTAL_BYTES,
                    recurrent_state_per_seq_bytes: Optional[int] = None,
                    ) -> List[Manifest]:
    """The DEFAULT topology for one (set, rank) cell on ``dims`` (module docstring;
    the 2026-10-01 ruling): ``[S1cc, S1d]`` for ``wo`` (the carriers ride S1cc),
    ``[S1cc, S1d, S2]`` for every other set. ``S1cc`` = control + constitution +
    zero at cap ``rank_cap(r)``; ``S1d`` = difference + zero at cap ``rank_cap(2r)``;
    each server's ``moe_lora_rank`` is its adapters' largest routed rank (None on
    a dense model). The memory arguments are accepted for signature compatibility
    (callers pass the plan's budget) — no VRAM-fit fallback is generated any more:
    the split IS the smaller server, and the probe's launches measure whether it
    fits."""
    if set_id not in MODULE_SETS:
        raise ManifestError(f"unknown module set {set_id!r}; the menu is {sorted(MODULE_SETS)}")
    rank = int(rank)
    if rank < 1:
        raise ManifestError(f"rank must be >= 1, got {rank}")
    modules = MODULE_SETS[set_id]
    cap_cc = rank_cap(rank)
    cap_d = rank_cap(2 * rank)
    target = target_class_for(dims, modules)
    control, constitution, difference = _cells(modules, rank, dims)
    cc_adapters: Tuple[AdapterSlot, ...] = (control, constitution, _zero(modules, cap_cc, dims))
    d_adapters: Tuple[AdapterSlot, ...] = (difference, _zero(modules, cap_d, dims))
    experts_note = (("routed experts wrapped: expert buffers E x moe_lora_rank x (2d + 3I) x 2 B per MoE block and slot (D394)",)
                    if EXPERTS_TARGET in target else ())
    if set_id == "wo":
        cc_adapters = cc_adapters + _carriers()
        s1cc = Manifest(name=f"wo_r{rank}_{TOPOLOGY_S1CC}", set_id="wo", rank=rank, topology=TOPOLOGY_S1CC,
                        adapters=cc_adapters, target_class=target, max_loras=len(cc_adapters), max_lora_rank=cap_cc,
                        notes=("the single-rank server: control + constitution + zero + the four carriers, all o_proj",))
        s1d = Manifest(name=f"wo_r{rank}_{TOPOLOGY_S1D}", set_id="wo", rank=rank, topology=TOPOLOGY_S1D,
                       adapters=d_adapters, target_class=target, max_loras=len(d_adapters), max_lora_rank=cap_d,
                       notes=("the difference server: the rank-2r difference + zero at cap >= 2r",))
        return [s1cc, s1d]
    s1cc = Manifest(name=f"{set_id}_r{rank}_{TOPOLOGY_S1CC}", set_id=set_id, rank=rank, topology=TOPOLOGY_S1CC,
                    adapters=cc_adapters, target_class=target, max_loras=len(cc_adapters), max_lora_rank=cap_cc,
                    notes=("the single-rank server: control + constitution + zero at cap >= r",) + experts_note
                    + (() if EXPERTS_TARGET in target else ("carriers target o_proj only, not this class: they ride S2",)))
    s1d = Manifest(name=f"{set_id}_r{rank}_{TOPOLOGY_S1D}", set_id=set_id, rank=rank, topology=TOPOLOGY_S1D,
                   adapters=d_adapters, target_class=target, max_loras=len(d_adapters), max_lora_rank=cap_d,
                   notes=("the difference server: the rank-2r difference + zero at cap >= 2r",) + experts_note)
    return [s1cc, s1d, build_s2_manifest(set_id, rank)]


#: the note every union manifest carries (the reader of a record tells the union from a legacy ``single`` by shape alone —
#: they ARE the same shape; the note says why it was planned)
UNION_NOTE = ("the served launch class (the probe owner's ruling of 2026-10-11): every exported block of a steered standup + "
              "none__lora0 in ONE engine — control r + constitution r + difference 2r + zero at cap >= 2r"
              " — certifies the CLASS a steered standup serves at this window (geometry.least_admitted_by_window: rank cap >= and "
              "slots >= at the SAME window); the roles stay certified by S1cc / S1d")


def build_union_manifest(dims: Any, *, set_id: str, rank: int) -> Manifest:
    """The pair's UNION of its S1 servers — the class a steered native standup of this (set, rank) SERVES (module docstring,
    "THE SERVED LAUNCH CLASS"): the three cells + the zero adapter at ``rank_cap(2r)``, plus the four rank-1 ``o_proj``
    carriers for ``wo`` (8 slots at cap >= 2r — control r4, constitution r4, difference r8, zero r8, 4 carriers = the
    ``(8, 8)`` the battery's standups serve); the cells + zero alone for every other set (4 slots; the carriers ride S2).
    Topology :data:`TOPOLOGY_SINGLE` — the legacy one-server shape, byte for byte for ``wo`` — so a reader orders it after
    ``S1cc`` / ``S1d`` and a legacy record's own ``single`` launches already cover it. Each server's ``moe_lora_rank`` is
    derived from its adapters as everywhere (None on a dense model). Never emitted by :func:`build_manifests` (the split
    servers certify the roles); the probe plans it beside them and the reopen's extension adds it where a record's planned
    launches do not dominate it."""
    if set_id not in MODULE_SETS:
        raise ManifestError(f"unknown module set {set_id!r}; the menu is {sorted(MODULE_SETS)}")
    rank = int(rank)
    if rank < 1:
        raise ManifestError(f"rank must be >= 1, got {rank}")
    modules = MODULE_SETS[set_id]
    cap = rank_cap(2 * rank)
    target = target_class_for(dims, modules)
    adapters: Tuple[AdapterSlot, ...] = _cells(modules, rank, dims) + (_zero(modules, cap, dims),)
    notes: Tuple[str, ...] = (UNION_NOTE,)
    if set_id == "wo":
        adapters = adapters + _carriers()
        notes = notes + ("the four residual carriers ride inside it (o_proj): 8 slots",)
    else:
        notes = notes + ("the carriers target o_proj only, not this class: they ride S2 (4 slots here)",)
        if EXPERTS_TARGET in target:
            notes = notes + ("routed experts wrapped: expert buffers at the manifest's moe_lora_rank (D394)",)
    return Manifest(name=f"{set_id}_r{rank}_{TOPOLOGY_SINGLE}", set_id=set_id, rank=rank, topology=TOPOLOGY_SINGLE,
                    adapters=adapters, target_class=target, max_loras=len(adapters), max_lora_rank=cap, notes=notes)


# --------------------------------------------------------------------------
# the bytes law
# --------------------------------------------------------------------------

def _stacked_bytes_per_slot(dims: Any, target_class: Sequence[str], R: int,
                            routed_R: Optional[int] = None) -> Dict[str, int]:
    """Per-slot LoRA buffer bytes per WRAPPED family, from the stacked
    buffer shapes (module docstring), bf16: the dense families at ``R`` (the
    cap), the routed-expert family at ``routed_R`` (``R`` when None —
    vLLM's own sizing, which every record before 2026-10-01 measured)."""
    R = int(R)
    routed_R = int(R if routed_R is None else routed_R)
    d = int(dims.hidden)
    out: Dict[str, int] = {}
    attn = _attention_modules(target_class)
    if any(m in attn for m in ("q_proj", "k_proj", "v_proj")):
        # the packed qkv wrapper is ONE module: one A per slice (3), B per slice
        slices = (int(dims.q_out), int(dims.kv_out), int(dims.kv_out))
        per_layer = 3 * R * d + R * sum(slices)
        out["attention_qkv_packed"] = per_layer * int(dims.attention_layers) * BF16_BYTES
    if "o_proj" in attn:
        per_layer = R * int(dims.o_in) + int(dims.hidden) * R
        out["attention_o_proj"] = per_layer * int(dims.attention_layers) * BF16_BYTES
    mlp = _mlp_modules(target_class)
    gate_up = "gate_proj" in mlp or "up_proj" in mlp
    down = "down_proj" in mlp
    if mlp and int(getattr(dims, "moe_layers", 0) or 0):
        I_s = int(dims.shared_intermediate)
        per_layer = 0
        if gate_up:   # packed gate_up: 2 slices, one A each
            per_layer += 2 * R * d + 2 * I_s * R
        if down:
            per_layer += R * I_s + d * R
        out["shared_expert"] = per_layer * int(dims.moe_layers) * BF16_BYTES
    if mlp and int(getattr(dims, "dense_mlp_layers", 0) or 0):
        I_d = int(dims.dense_intermediate)
        per_layer = 0
        if gate_up:   # packed gate_up: 2 slices, one A each
            per_layer += 2 * R * d + 2 * I_d * R
        if down:
            per_layer += R * I_d + d * R
        out["dense_mlp"] = per_layer * int(dims.dense_mlp_layers) * BF16_BYTES
    if EXPERTS_TARGET in target_class:
        E = int(dims.num_experts)
        I = int(dims.moe_intermediate)
        # w13 A [E, R, d] (one shared slice) + w13 B [E, 2I, R] + w2 A [E, R, I] + w2 B [E, d, R] — at the ROUTED rank
        per_layer = E * routed_R * (d + 2 * I + I + d)
        out["routed_experts"] = per_layer * int(dims.moe_layers) * BF16_BYTES
    return out


def expected_lora_bytes(manifest: Manifest, R: int, dims: Any, *,
                        moe_lora_rank: Optional[int] = None) -> Dict[str, Any]:
    """The expected STACKED LoRA buffer bytes of ``manifest``'s engine at cap
    ``R``: per family per slot, the per-slot total, × ``max_loras``. The dense
    families are priced at ``R``, the routed-expert family at ``moe_lora_rank``
    (default: the manifest's own). ``per_slot_mib_per_R`` divides each
    family by the rank IT was priced at. The one formula the probe asserts the
    worker-reported bytes against (5 %)."""
    routed = manifest.moe_lora_rank if moe_lora_rank is None else int(moe_lora_rank)
    per_slot = _stacked_bytes_per_slot(dims, manifest.target_class, R, routed)
    slot_total = sum(per_slot.values())
    rank_of = {k: (int(routed) if (k == "routed_experts" and routed is not None) else int(R)) for k in per_slot}
    return {"R": int(R), "moe_lora_rank": (int(routed) if routed is not None else None),
            "max_loras": int(manifest.max_loras), "dtype_bytes": BF16_BYTES,
            "target_class": list(manifest.target_class),
            "per_slot": dict(per_slot), "per_slot_total": int(slot_total),
            "total": int(slot_total * manifest.max_loras),
            "per_slot_mib_per_R": {k: v / rank_of[k] / 2**20 for k, v in per_slot.items()},
            "rank_by_family": rank_of}


def buffer_bytes_within(expected: int, measured: int, tolerance: float = 0.05) -> Dict[str, Any]:
    """The 5 % assertion as a record: ``{expected, measured, rel_err, pass}``."""
    expected = int(expected)
    measured = int(measured)
    rel = (abs(measured - expected) / expected) if expected else (0.0 if measured == 0 else math.inf)
    return {"expected": expected, "measured": measured, "rel_err": rel,
            "tolerance": tolerance, "pass": rel <= tolerance}


# --------------------------------------------------------------------------
# strides, ceilings, KV, admission
# --------------------------------------------------------------------------

def executing_strides(dims: Any, target_class: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """``{wrapper: {"in", "out"}}`` for every EXECUTION wrapper the class
    restricts vLLM to (the packed wrapper's ``out`` is ``sum(output_slices)``
    — ``worker_reports.py:1353``)."""
    d = int(dims.hidden)
    out: Dict[str, Dict[str, int]] = {}
    attn = _attention_modules(target_class)
    if any(m in attn for m in ("q_proj", "k_proj", "v_proj")):
        out["self_attn.qkv_proj"] = {"in": d, "out": int(dims.q_out) + 2 * int(dims.kv_out)}
    if "o_proj" in attn:
        out["self_attn.o_proj"] = {"in": int(dims.o_in), "out": d}
    mlp = _mlp_modules(target_class)
    gate_up = "gate_proj" in mlp or "up_proj" in mlp
    if mlp and int(getattr(dims, "moe_layers", 0) or 0):
        I_s = int(dims.shared_intermediate)
        if gate_up:
            out["mlp.shared_expert.gate_up_proj"] = {"in": d, "out": 2 * I_s}
        if "down_proj" in mlp:
            out["mlp.shared_expert.down_proj"] = {"in": I_s, "out": d}
    if mlp and int(getattr(dims, "dense_mlp_layers", 0) or 0):
        I_d = int(dims.dense_intermediate)
        if gate_up:
            out["mlp.gate_up_proj"] = {"in": d, "out": 2 * I_d}
        if "down_proj" in mlp:
            out["mlp.down_proj"] = {"in": I_d, "out": d}
    if EXPERTS_TARGET in target_class:
        I = int(dims.moe_intermediate)
        out["mlp.experts.w13"] = {"in": d, "out": 2 * I}
        out["mlp.experts.w2"] = {"in": I, "out": d}
    return out


def packed_stride_ceiling(dims: Any, target_class: Sequence[str]) -> Dict[str, Any]:
    """``{D, widest, ceiling}``: the widest executing stride of the class and
    the int32 batched-token ceiling ``(2^31-1) // D`` (on the Qwen3.5-4B dims:
    Q/K/V adapted -> 8192 + 2 × 1024 -> 209,715 tokens; ``wo`` -> 4,096 ->
    524,287)."""
    strides = executing_strides(dims, target_class)
    if not strides:
        raise ManifestError(f"target class {list(target_class)} wraps nothing on these dims")
    widest_name, widest = max(strides.items(), key=lambda kv: (max(kv[1]["in"], kv[1]["out"]), kv[0]))
    D = int(max(widest["in"], widest["out"]))
    return {"D": D, "widest": widest_name, "ceiling": lora_kernel_index_ceiling(D),
            "limit": LORA_KERNEL_INDEX_LIMIT, "strides": strides}


def kv_bytes_per_token(dims: Any) -> int:
    """Per-token KV bytes in bf16: attention blocks × (K + V) × kv width."""
    return int(dims.attention_layers) * 2 * int(dims.kv_out) * BF16_BYTES


def mlp_intermediate_elements(dims: Any) -> Dict[str, Any]:
    """The MLP-intermediate activation ONE token holds while ONE block's MLP
    executes, in elements, by architecture (the padded preflight's warmup term
    is ``S × L × elements × 2 B``): a dense block one
    ``dense_intermediate``-wide activation; an MoE block one per ACTIVE
    expert (``active_experts × moe_intermediate``) plus the shared expert's
    (``shared_intermediate``); a model with both kinds of block takes the
    wider, since one block is live at a time. Zero blocks of a kind
    contribute nothing; a dims with neither kind yields ``elements`` None.
    ``{elements, moe_block, dense_block, terms}``."""
    moe_block = dense_block = None
    if int(getattr(dims, "moe_layers", 0) or 0):
        moe_block = (int(getattr(dims, "active_experts", 1) or 1) * int(dims.moe_intermediate)
                     + int(getattr(dims, "shared_intermediate", 0) or 0))
    if int(getattr(dims, "dense_mlp_layers", 0) or 0):
        dense_block = int(dims.dense_intermediate)
    widths = [w for w in (moe_block, dense_block) if w]
    return {"elements": (max(widths) if widths else None), "moe_block": moe_block, "dense_block": dense_block,
            "terms": {"moe_layers": int(getattr(dims, "moe_layers", 0) or 0), "active_experts": int(getattr(dims, "active_experts", 1) or 1),
                      "moe_intermediate": int(getattr(dims, "moe_intermediate", 0) or 0),
                      "shared_intermediate": int(getattr(dims, "shared_intermediate", 0) or 0),
                      "dense_mlp_layers": int(getattr(dims, "dense_mlp_layers", 0) or 0),
                      "dense_intermediate": int(getattr(dims, "dense_intermediate", 0) or 0)}}


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def gated_deltanet_state_per_seq_bytes(config: Any, *, dtype_bytes: int = BF16_BYTES) -> int:
    """The per-SEQUENCE recurrent state of a Gated-DeltaNet hybrid (constant in
    ``L``), derived from the checkpoint's HF TEXT config — a multimodal top-level
    config's ``text_config`` is read when it has one — never a typed per-model
    constant: ``linear blocks × (the SSM state linear_num_value_heads ×
    linear_key_head_dim × linear_value_head_dim + the conv state
    (linear_conv_kernel_dim − 1) × (2 × linear_num_key_heads × linear_key_head_dim
    + linear_num_value_heads × linear_value_head_dim)) × dtype_bytes``. The block
    count is ``layer_types.count("linear_attention")`` when the config states its
    layer types, else ``num_hidden_layers − num_hidden_layers //
    full_attention_interval``. A config with none of :data:`GDN_CONFIG_FIELDS` (a
    dense transformer) yields 0; one with SOME of them refuses by name.
    ``dtype_bytes`` is the state dtype the engine allocates (bf16 here; a launch
    under ``--mamba-ssm-cache-dtype float32`` widens the SSM term — the caller
    states what it launched). The probe's plan carries the result as the
    budget's ``recurrent_state_per_seq_bytes``."""
    cfg = config
    inner = _cfg_get(cfg, "text_config")
    if inner is not None and _cfg_get(inner, GDN_CONFIG_FIELDS[0]) is not None:
        cfg = inner
    vals = {name: _cfg_get(cfg, name) for name in GDN_CONFIG_FIELDS}
    present = {k for k, v in vals.items() if v is not None}
    if not present:
        return 0
    if present != set(GDN_CONFIG_FIELDS):
        raise ManifestError(f"a Gated-DeltaNet config states {sorted(present)} but not "
                            f"{sorted(set(GDN_CONFIG_FIELDS) - present)}; the recurrent state cannot be sized")
    for k, v in vals.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ManifestError(f"{k}={v!r} is not a positive int")
    layer_types = _cfg_get(cfg, "layer_types")
    if layer_types:
        n_linear = sum(1 for t in layer_types if str(t) == "linear_attention")
    else:
        n_layers = int(_cfg_get(cfg, "num_hidden_layers", 0) or 0)
        interval = int(_cfg_get(cfg, "full_attention_interval", 0) or 0)
        if n_layers < 1 or interval < 1:
            raise ManifestError("a Gated-DeltaNet config states neither layer_types nor "
                                "num_hidden_layers + full_attention_interval; the linear-block count cannot be read")
        n_linear = n_layers - n_layers // interval
    ssm = vals["linear_num_value_heads"] * vals["linear_key_head_dim"] * vals["linear_value_head_dim"]
    conv_dim = 2 * vals["linear_num_key_heads"] * vals["linear_key_head_dim"] + vals["linear_num_value_heads"] * vals["linear_value_head_dim"]
    conv = (vals["linear_conv_kernel_dim"] - 1) * conv_dim
    return int(n_linear) * (int(ssm) + int(conv)) * int(dtype_bytes)


def kv_bytes_per_seq(dims: Any, L: int, *, recurrent_state_bytes: int = 0) -> int:
    """Per-SEQUENCE bytes at window ``L``: KV × L plus a constant per-sequence
    recurrent state when the architecture carries one
    (:func:`gated_deltanet_state_per_seq_bytes`; 0 for a dense transformer)."""
    return kv_bytes_per_token(dims) * int(L) + int(recurrent_state_bytes)


def admitted_max_num_seqs(dims: Any, target_class: Sequence[str], L: int, *,
                          kv_budget_bytes: Optional[float] = None,
                          recurrent_state_bytes: int = 0,
                          requested_S: Optional[int] = None,
                          batched_token_budget: Optional[int] = None) -> Dict[str, Any]:
    """The S the standup would ADMIT at window ``L``: the packed-stride
    ceiling ``ceiling // L`` (never clamped by the composer — the caller
    lowers), the KV-derived S when a budget is given, the min of both with
    ``requested_S``, and the PREFILL term
    ``floor(batched_token_budget / L)``: the harness's no-chunking invariant
    makes an engine prefill its whole ``max_num_batched_tokens = S x L`` in
    one batch, an activation the KV term never priced (a plan whose D = 2,048
    sets kept S = 21 at L = 49,152 — 1.03 M tokens — died in
    ``_initialize_kv_caches`` while 262,144 was the largest batch that ever
    came up). ``batched_token_budget`` is a MEASURED number (the serving
    probe's record floor, or a site geometry overlay's ``ModelBudget``), never
    a typed constant; None applies no cap and the record says so
    (``budget_S`` None). The record carries ``budget_S``, ``batched_tokens``
    (= admitted S x L) and ``binding`` names ``batched_tokens`` when that term
    bound. ``S < 1`` is recorded (the pair is excluded with
    ``ceiling_S<1_at_L=<L>``), never raised."""
    L = int(L)
    ceil = packed_stride_ceiling(dims, target_class)
    stride_S = ceil["ceiling"] // L
    rec: Dict[str, Any] = {"L": L, "D": ceil["D"], "widest": ceil["widest"], "ceiling": ceil["ceiling"],
                           "stride_S": int(stride_S), "kv_S": None,
                           "kv_per_seq_bytes": kv_bytes_per_seq(dims, L, recurrent_state_bytes=recurrent_state_bytes),
                           "requested_S": requested_S,
                           "batched_token_budget": (int(batched_token_budget) if batched_token_budget is not None else None),
                           "budget_S": None}
    admitted = stride_S
    if kv_budget_bytes is not None:
        kv_S = int(kv_budget_bytes // rec["kv_per_seq_bytes"]) if rec["kv_per_seq_bytes"] else 0
        rec["kv_S"] = kv_S
        admitted = min(admitted, kv_S)
    if requested_S is not None:
        admitted = min(admitted, int(requested_S))
    binding = ("stride" if admitted == stride_S else
               "kv" if rec["kv_S"] is not None and admitted == rec["kv_S"] else "requested")
    if batched_token_budget is not None:
        budget_S = int(batched_token_budget) // L
        rec["budget_S"] = int(budget_S)
        if budget_S < admitted:
            admitted = budget_S
            binding = "batched_tokens"
    rec["admitted_S"] = int(admitted)
    rec["batched_tokens"] = int(admitted) * L
    rec["binding"] = binding
    return rec


def estimated_vram_bytes(manifest: Manifest, R: int, dims: Any, *,
                         base_weights_bytes: float = DEFAULT_BASE_WEIGHTS_BYTES,
                         kv_per_seq_bytes: Optional[float], S: Optional[int],
                         kv_budget_bytes: Optional[float] = None,
                         graphs_activations_bytes: float = DEFAULT_GRAPHS_ACTIVATIONS_BYTES,
                         gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
                         total_vram_bytes: float = H200_TOTAL_BYTES,
                         batched_tokens: Optional[int] = None,
                         prefill_bytes_per_token: Optional[float] = None) -> Dict[str, Any]:
    """The VRAM a launch of ``manifest`` at cap ``R`` needs: base weights +
    the LoRA buffers (:func:`expected_lora_bytes`) + graphs/activations +
    the KV budget (``S × kv_per_seq_bytes``, or ``kv_budget_bytes`` directly)
    + the PREFILL activation of the no-chunking batch, ``batched_tokens ×
    prefill_bytes_per_token``, when BOTH are given (the constant
    ``graphs_activations_bytes`` never scaled with ``S x L``, which is how a
    1.03 M-token prefill was estimated to fit). With no measured per-token
    cost the prefill is reported as ``batched_tokens`` with
    ``prefill_activations`` None (not priced — the admission's
    ``batched_token_budget`` is what bounds it), never a typed constant;
    against ``gpu_memory_utilization × total_vram_bytes``."""
    lora = expected_lora_bytes(manifest, R, dims)
    if kv_budget_bytes is None:
        if kv_per_seq_bytes is None or S is None:
            raise ManifestError("estimated_vram_bytes needs kv_budget_bytes, or kv_per_seq_bytes and S")
        kv_budget_bytes = float(kv_per_seq_bytes) * int(S)
    prefill: Optional[float] = None
    if batched_tokens is not None and prefill_bytes_per_token is not None:
        prefill = float(int(batched_tokens)) * float(prefill_bytes_per_token)
    total = (float(base_weights_bytes) + lora["total"] + float(graphs_activations_bytes) + float(kv_budget_bytes)
             + (prefill or 0.0))
    usable = float(gpu_memory_utilization) * float(total_vram_bytes)
    return {"base_weights": float(base_weights_bytes), "lora_buffers": int(lora["total"]),
            "graphs_activations": float(graphs_activations_bytes), "kv": float(kv_budget_bytes),
            "batched_tokens": (int(batched_tokens) if batched_tokens is not None else None),
            "prefill_activations": prefill, "prefill_bytes_per_token": prefill_bytes_per_token,
            "S": S, "total": total, "usable": usable, "total_vram": float(total_vram_bytes),
            "gpu_memory_utilization": float(gpu_memory_utilization), "fits": total <= usable,
            "headroom": usable - total}


def trainable_params(dims: Any, set_id: str, rank: int) -> int:
    """``lora_modules.formula_parameter_count`` on the pair (exact for the real dims)."""
    return int(formula_parameter_count(dims, set_id, int(rank)))


def pair_per_expert_rank(dims: Any, set_id: str, rank: int) -> Optional[Dict[str, int]]:
    """``{"shared_expert": r, "routed_expert": r_e}`` when the set reaches
    routed experts on ``dims``, else None."""
    if not adapts_routed_experts(dims, MODULE_SETS[set_id]):
        return None
    return per_expert_rank_map(int(rank), int(getattr(dims, "active_experts", 1) or 1))


def topology_label(manifests: Sequence[Manifest]) -> str:
    """The pair's topology string over the manifests that CERTIFIED it: ``single``
    for the legacy one-server ``wo``; ``S1S2`` for the legacy four-slot server +
    S2; ``S1abcS2`` when the legacy fallback servers (and no S1) certified;
    otherwise the present labels joined in :data:`TOPOLOGY_ORDER` — ``S1ccS1d``
    (``wo``), ``S1ccS1dS2`` (every other set), and a MIXED record's e.g.
    ``S1S1ccS1dS2`` when some windows were certified by the legacy server."""
    tops = {m.topology for m in manifests}
    if tops == {TOPOLOGY_SINGLE}:
        return TOPOLOGY_SINGLE
    if tops <= set(S1_FALLBACKS) | {TOPOLOGY_S2} and any(t in S1_FALLBACKS for t in tops):
        return "S1abcS2"
    if tops <= {TOPOLOGY_S1, TOPOLOGY_S2}:
        return TOPOLOGY_LABEL_TWO_SERVER
    known = [t for t in TOPOLOGY_ORDER if t in tops]
    other = sorted(t for t in tops if t not in TOPOLOGY_ORDER)
    return "".join(known + other)


__all__ = [
    "LEGAL_LORA_RANK_CAPS", "RANK_CAP_LADDER", "PRODUCTION_WINDOWS", "EXPERTS_TARGET", "GDN_CONFIG_FIELDS",
    "H200_TOTAL_BYTES", "S1_FALLBACK_BY_ROLE", "S1_FALLBACKS",
    "S2_RANK_CAP", "TOPOLOGY_SINGLE", "TOPOLOGY_S1", "TOPOLOGY_S2", "TOPOLOGY_S1CC", "TOPOLOGY_S1D", "TOPOLOGY_ORDER",
    "ROLE_CONTROL", "ROLE_CONSTITUTION", "ROLE_DIFFERENCE", "GENUINE_ROLES", "ROLES_REQUIRED",
    "AdapterSlot", "Manifest", "ManifestError", "difference_per_expert_rank",
    "rank_cap", "target_class_for", "adapts_routed_experts", "build_manifests", "build_s2_manifest", "build_union_manifest",
    "UNION_NOTE",
    "expected_lora_bytes", "buffer_bytes_within", "executing_strides", "packed_stride_ceiling",
    "kv_bytes_per_token", "kv_bytes_per_seq", "mlp_intermediate_elements", "gated_deltanet_state_per_seq_bytes",
    "admitted_max_num_seqs", "estimated_vram_bytes", "trainable_params", "pair_per_expert_rank", "topology_label",
]
