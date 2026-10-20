"""ONE builder for the fittable objects of the pipeline — the LoRA
(:class:`~sisteer.extraction.multilayer.MultiLayerLoRAObject`) and the
ReLU steer (:class:`~sisteer.extraction.psr.PSRObject`) — shared by the
production trainers (``methods._train_lora_ce`` / ``_train_psr_ce``), the
HPO trial runner (``scripts/tune_arm.py``'s ``CESoloAdapter.make_object``),
the gradient-norm reconnaissance and the memory probe.

Before this module the search runner constructed the objects DIRECTLY and
refused any ``lora_rank`` while the production trainer took ``lora_rank``
alone, so the architecture and initialisation axes of the LoRA search (FD
6, FD 7) had no single entry point. Now every caller says WHAT it wants
(``object_kind``, ``lora_rank``, ``lora_modules``, ``per_expert_rank``,
``init_scheme``, ``seed``) and gets the identical object: the same seed
and architecture give bit-identical initial tensors in a search trial, a
production fit and the constitution/control twins of a seed
(``tests/test_object_factory_parity.py``).

The signature is keyword-only with NO ``**kw``: a kwarg this factory does
not know raises ``TypeError`` (a design that names an axis the object
cannot express must fail at the call, never be silently dropped).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from ..models import num_layers
from .lora_modules import (MLP_MODULES, PAPER_LORA_MODULES, PAPER_LORA_RANK,
                           LoRAModuleError, layers_by_module, normalise_modules,
                           per_expert_rank_map, set_id_for)

#: The two object classes the pipeline fits, spelled as the registry does
#: (``vocab.OBJECT_KINDS``), plus the legacy arm / kind spellings callers use.
OBJECT_KINDS: tuple[str, ...] = ("lora", "relu_steer")
KIND_ALIASES: dict[str, str] = {
    "lora": "lora", "lora_all": "lora", "lora_r4_ce": "lora", "lora_r4": "lora",
    "relu_steer": "relu_steer", "psr": "relu_steer", "psr_ce_all": "relu_steer",
    "psr_kl_all": "relu_steer",
}


class ObjectFactoryError(ValueError):
    """A request the factory refuses by name (unknown kind, a LoRA axis on
    the ReLU steer, an init scheme outside the vocabulary, …)."""


def canonical_kind(object_kind: str) -> str:
    kind = KIND_ALIASES.get(str(object_kind))
    if kind is None:
        raise ObjectFactoryError(
            f"unknown object_kind {object_kind!r}; the factory builds "
            f"{OBJECT_KINDS} (aliases: {sorted(KIND_ALIASES)})")
    return kind


def build_object(model, *, object_kind: str, lora_rank: int | None = None,
                 lora_modules: Iterable[str] | str | None = None,
                 per_expert_rank: int | Mapping | None = None,
                 init_scheme: str = "zero", seed: int = 0,
                 layers: Sequence[int] | None = None):
    """Build the untrained object for ``object_kind`` on ``model``.

    LoRA: ``lora_rank`` (default the paper's 4), ``lora_modules`` (a set id
    or list; default the paper's ``["o_proj"]`` — that default path is the
    class's legacy shape, bit-identical to the pre-search object),
    ``per_expert_rank`` (optional — an int or the registry's map; the
    object validates it against ``max(1, r // active)``),
    ``init_scheme`` (``"zero"`` = the paper's B = 0; ``"random_lecun_1e-3"``
    = FD 7's run override), ``seed`` (the fit seed — seeds A always and B
    under the random scheme), ``layers`` (an explicit block list; default
    every eligible block).

    ReLU steer: every ``lora_*`` argument must be None (refused by name);
    ``layers`` defaults to all blocks; ``train_direction=True`` as the
    ``psr_ce_all`` arm has always built it; ``init_scheme``/``seed`` per FD 7.
    """
    kind = canonical_kind(object_kind)
    seed = int(seed)
    # Every refusal below the factory (an init scheme outside the vocabulary,
    # a per_expert_rank that contradicts FD 6, an o_proj-only object handed a
    # per-expert rank, a set that resolves to nothing) is the OBJECT's own
    # ValueError, re-raised as the factory's typed error with its wording.
    if kind == "relu_steer":
        stray = {k: v for k, v in (("lora_rank", lora_rank), ("lora_modules", lora_modules),
                                   ("per_expert_rank", per_expert_rank)) if v is not None}
        if stray:
            raise ObjectFactoryError(
                f"relu_steer takes no LoRA axes; got {stray} — the ReLU steer's "
                "architecture is the paper's (FD 6: no rank/module-set search)")
        from .psr import PSRObject

        layer_list = (list(range(num_layers(model))) if layers is None
                      else [int(l) for l in layers])
        try:
            return PSRObject(model, layer_list, train_direction=True,
                             init_scheme=init_scheme, seed=seed)
        except ValueError as e:
            raise ObjectFactoryError(str(e)) from e
    from .multilayer import MultiLayerLoRAObject

    rank = PAPER_LORA_RANK if lora_rank is None else int(lora_rank)
    if rank < 1:
        raise ObjectFactoryError(f"lora_rank must be >= 1, got {lora_rank!r}")
    try:
        return MultiLayerLoRAObject(
            model, layers=(list(layers) if layers is not None else None),
            seed=seed, rank=rank,
            modules=PAPER_LORA_MODULES if lora_modules is None else lora_modules,
            per_expert_rank=per_expert_rank, init_scheme=init_scheme)
    except ValueError as e:   # LoRAModuleError is a ValueError too
        raise ObjectFactoryError(str(e)) from e


def describe_object(obj) -> dict:
    """The registry-facing description of a built object (the
    ``params_json`` names): ``object_kind``, ``lora_rank``, ``lora_modules``,
    ``lora_set_id``, ``per_expert_rank``, ``init_scheme``,
    ``trainable_params``, ``layers_by_module``. ReLU steers report their
    layer list under ``layers_by_module["residual"]``."""
    describe = getattr(obj, "describe", None)
    if not callable(describe):
        raise ObjectFactoryError(
            f"cannot describe an object of kind {getattr(obj, 'kind', None)!r} "
            "(no describe(): not a LoRA object or a ReLU steer)")
    return describe()


def architecture_params(*, object_kind: str, lora_rank: int | None,
                        lora_modules: Iterable[str] | str | None,
                        active_experts: int | None, init_scheme: str = "zero") -> dict:
    """The architecture half of a trial's / object's ``params_json`` WITHOUT
    building anything (the design writer and the table need it before a
    model is loaded): ``lora_rank``, ``lora_modules`` (sorted), ``lora_set_id``,
    ``per_expert_rank`` (the FD 6 map when ``active_experts`` is given and
    the set reaches the FFN, else None), ``init_scheme``."""
    kind = canonical_kind(object_kind)
    if kind == "relu_steer":
        return {"object_kind": kind, "lora_rank": None, "lora_modules": None,
                "lora_set_id": None, "per_expert_rank": None,
                "init_scheme": str(init_scheme)}
    rank = PAPER_LORA_RANK if lora_rank is None else int(lora_rank)
    modules = normalise_modules(PAPER_LORA_MODULES if lora_modules is None else lora_modules)
    reaches_ffn = any(m in MLP_MODULES for m in modules)
    per = (per_expert_rank_map(rank, int(active_experts))
           if (reaches_ffn and active_experts) else None)
    return {"object_kind": kind, "lora_rank": rank, "lora_modules": list(modules),
            "lora_set_id": set_id_for(modules), "per_expert_rank": per,
            "init_scheme": str(init_scheme)}


__all__ = ["OBJECT_KINDS", "KIND_ALIASES", "ObjectFactoryError", "canonical_kind",
           "build_object", "describe_object", "architecture_params",
           "layers_by_module"]
