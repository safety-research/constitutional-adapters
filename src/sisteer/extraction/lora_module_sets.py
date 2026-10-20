"""The LoRA module-set VOCABULARY — a stdlib leaf: no torch, no transformers, importable under ``python3 -S``.

:mod:`sisteer.extraction.lora_modules` is the module-set design (its docstring is the design note) and
RE-EXPORTS every name defined here, so ``from .lora_modules import MODULE_SETS`` and ``lora_modules.set_id_for`` are
unchanged and the two modules hold ONE object per name (identity, not a copy). This leaf exists for the readers that need
only the vocabulary: the steered-padding preflight's module-set selection (``padding_preflight.selected_module_set``, run by
the post-standup geometry pin inside the suite's CPU/stdlib venv, which has no transformers) — ``lora_modules`` imports torch
and ``..models`` (transformers) at import, which that reader must not need. The re-export identity is pinned by
tests/test_lora_modules.py, the torch-/transformers-free pin by tests/test_padding_front_door_view_families.py.
(``scripts/tune_arm.py`` still carries its own torch-free MIRROR of ``MODULE_SETS`` from before this leaf existed;
tests/test_hpo_guard_discrete_axes.py pins the two tables equal.)
"""

from __future__ import annotations

from typing import Iterable

#: The module-set vocabulary: set id -> sorted short HF names.
MODULE_SETS: dict[str, tuple[str, ...]] = {
    "wo": ("o_proj",),
    "qkvo": ("k_proj", "o_proj", "q_proj", "v_proj"),
    "attn_mlp": ("down_proj", "gate_proj", "k_proj", "o_proj", "q_proj",
                 "up_proj", "v_proj"),
    "mlp": ("down_proj", "gate_proj", "up_proj"),
    "down": ("down_proj",),
}

ATTENTION_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
KNOWN_MODULES: frozenset[str] = frozenset(ATTENTION_MODULES + MLP_MODULES)


class LoRAModuleError(ValueError):
    """A module set the vocabulary does not define, a set that resolves to
    nothing on the model, or an expert factor request the fused layout
    cannot express."""


def normalise_modules(modules: Iterable[str] | str) -> tuple[str, ...]:
    """The SORTED, de-duplicated short-name tuple for ``modules`` — a set
    id (``"mlp"``), a comma-separated string, or any iterable of names.
    Unknown names refuse by name."""
    if isinstance(modules, str):
        if modules in MODULE_SETS:
            return MODULE_SETS[modules]
        parts = [m.strip() for m in modules.split(",") if m.strip()]
    else:
        parts = [str(m).strip() for m in modules]
    if not parts:
        raise LoRAModuleError("empty module set")
    unknown = sorted(set(parts) - KNOWN_MODULES)
    if unknown:
        raise LoRAModuleError(
            f"unknown LoRA target module(s) {unknown}; the vocabulary is "
            f"{sorted(KNOWN_MODULES)} (sets: {sorted(MODULE_SETS)})")
    return tuple(sorted(set(parts)))


def set_id_for(modules: Iterable[str] | str) -> str:
    """The set id (``wo``/``qkvo``/``attn_mlp``/``mlp``/``down``) of a
    module list, or the sorted ``+``-joined names for an ad-hoc set that
    is not one of the five (never refused: the registry spelling is the
    list, the id is a label)."""
    mods = normalise_modules(modules)
    for sid, members in MODULE_SETS.items():
        if members == mods:
            return sid
    return "+".join(mods)
