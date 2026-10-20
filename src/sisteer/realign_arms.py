"""Realignment-campaign arm registry.

One `RealignArm` per arm of the realign matrix, in the §0.2 order
(order-stable, so sbatch array indices decode deterministically), plus
`arm_model_spec` — the SINGLE place that resolves which WEIGHTS a cell
loads. Suite runners stay model-agnostic: they receive
`(model, tok, items, system, obj, scale)` and emit the same per-item
records for every arm; whether `model` is the base reference, the merged
organism, or a merged+realigned FT checkpoint is invisible below the
driver. The FT (`model_swap`) arms therefore keep the IDENTICAL
summary/record contract as the object arms — they just evaluate with
`obj=None` on swapped-in weights.

Arm kinds:
- "model": no intervention beyond which weights load (reference/baseline).
- "prompt": the SI text as system prompt on the merged organism.
- "object": a trained runtime intervention (hook) on the merged organism.
- "model_swap": REAL retrained weights (F1) — a saved LoRA adapter merged
  on top of the organism — evaluated `obj=None`.

The teacher axis ("self" = organism+SI, "external" = base-Instruct+SI)
governs both pos provenance and, for distill arms, the soft-KL teacher
logits (F7, `train_distill(teacher_model=...)`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .data.instructions import Instruction
from .data.pairs import Pair, pairs_to_distill_examples
from .extraction.distill import train_distill
from .extraction.sft import train_sft_lora
from .methods import VectorObject

TEACHER_SELF = "self"
TEACHER_EXTERNAL = "external"
_BOTH_TEACHERS = (TEACHER_SELF, TEACHER_EXTERNAL)


# -- trainers (methods.py signature: model, tok, instruction, pairs, *, ...) -----


def _train_distill_arm(model, tok, instruction: Instruction, pairs: list[Pair],
                       *, layer: int, teacher_model=None, log_fn=None,
                       **kw) -> VectorObject:
    """distill_* arms: train_distill(loss='soft') at `layer`. External
    cells pass base-Instruct as teacher_model (F7); self cells pass None
    (byte-identical to the S2 self-teacher path)."""
    exs = pairs_to_distill_examples(tok, pairs, instruction)
    sv = train_distill(model, tok, exs, layer, loss="soft",
                       teacher_model=teacher_model,
                       instruction_id=instruction.id, log_fn=log_fn, **kw)
    return VectorObject(sv)


def _train_ft_lora_arm(model, tok, instruction: Instruction, pairs: list[Pair],
                       *, out_dir, layer=None, teacher_model=None, log_fn=None,
                       batch_size: int = 2, seed: int = 0, **kw) -> Path:
    """ft_lora_* arms (F1): a REAL saved PEFT adapter; returns its path.
    `layer` is accepted for driver uniformity and unused — LoRA targets
    are module names, not a hook layer."""
    return train_sft_lora(model, tok, pairs, instruction,
                          teacher_model=teacher_model, out_dir=out_dir,
                          batch_size=batch_size, seed=seed, log_fn=log_fn,
                          **kw)


# -- the registry -----------------------------------------------------------------


@dataclass(frozen=True)
class RealignArm:
    name: str
    kind: Literal["model", "prompt", "object", "model_swap"]
    si_id: str | None  # instructions.py registry id (si_constitution / ...)
    teachers: tuple[str, ...]  # () for untrained arms
    trainer: Callable | None  # None for untrained / declared-only arms
    artifact_name: str | None  # per-cell artifact file/dir name


REALIGN_ARMS: dict[str, RealignArm] = {
    arm.name: arm
    for arm in [
        # Base Instruct, no adapter — doubles as the "perfectly undo the
        # EM adapter" oracle.
        RealignArm("reference", "model", None, (), None, None),
        # The merged organism, unrealigned.
        RealignArm("baseline", "model", None, (), None, None),
        # SI text as system prompt on the organism.
        RealignArm("prompted_constitution", "prompt", "si_constitution",
                   (), None, None),
        RealignArm("prompted_caa", "prompt", "si_caa", (), None, None),
        # Compiled L15 vectors (soft-KL distill) from rejection-sampled
        # pairs, per teacher source.
        RealignArm("distill_constitution", "object", "si_constitution",
                   _BOTH_TEACHERS, _train_distill_arm, "vector.safetensors"),
        RealignArm("distill_caa", "object", "si_caa", _BOTH_TEACHERS,
                   _train_distill_arm, "vector.safetensors"),
        # REAL-weight LoRA fine-tuning (F1): trained + saved adapters,
        # evaluated as a model swap with obj=None.
        RealignArm("ft_lora_constitution", "model_swap", "si_constitution",
                   _BOTH_TEACHERS, _train_ft_lora_arm, "adapter"),
        RealignArm("ft_lora_caa", "model_swap", "si_caa", _BOTH_TEACHERS,
                   _train_ft_lora_arm, "adapter"),
    ]
}


def arm_names() -> list[str]:
    """§0.2 order (dicts preserve insertion order) — the sbatch-array
    enumeration base."""
    return list(REALIGN_ARMS)


# -- model resolution: the one place that decides which weights a cell loads -----


@dataclass(frozen=True)
class ModelSpec:
    """What to load for a cell: a base checkpoint plus an ordered stack of
    PEFT adapters to merge (possibly empty). Feed straight into
    `load_model_and_tokenizer(spec.name, revision=spec.revision,
    dtype=spec.dtype, adapters=list(spec.adapters) or None)` — or call
    `spec.load()`."""

    name: str
    revision: str | None = None
    adapters: tuple[tuple[str, str | None], ...] = ()
    dtype: str | None = None

    def load(self, device: str | None = None, **kwargs):
        from .models import load_model_and_tokenizer

        return load_model_and_tokenizer(
            self.name,
            device=device,
            dtype=self.dtype,
            revision=self.revision,
            adapters=list(self.adapters) or None,
            **kwargs,
        )


def _organism_adapter(cfg: dict) -> tuple[str, str | None]:
    if "model_adapter" not in cfg:
        raise KeyError(
            "cfg has no 'model_adapter' — the organism arms need the EM "
            "adapter repo/path (+ 'model_adapter_revision') in the config"
        )
    return (cfg["model_adapter"], cfg.get("model_adapter_revision"))


def arm_model_spec(arm: RealignArm | str, cfg: dict) -> ModelSpec:
    """The SINGLE resolver of which weights a cell loads.

    - `reference` -> the base model, NO adapter.
    - every other model/prompt/object arm -> the merged organism
      (base + EM adapter): interventions and prompts act on the organism.
    - `model_swap` arms -> the realigned checkpoint: base + EM adapter +
      the cell's trained realignment adapter (all merged). The cell's
      trained-artifact path arrives as `cfg['ft_artifact']` — the driver
      injects it per cell (FT cells are per SI x teacher, so the path
      cannot be a static registry fact).

    Suite runners never branch on any of this: they get (model, tok, ...)
    plus `obj` (None for model/prompt/model_swap arms) and emit identical
    per-item records across arms.
    """
    if isinstance(arm, str):
        arm = REALIGN_ARMS[arm]
    base = ModelSpec(
        name=cfg["model"],
        revision=cfg.get("model_revision"),
        dtype=cfg.get("dtype"),
    )
    if arm.name == "reference":
        return base
    if arm.kind in ("model", "prompt", "object"):
        return ModelSpec(
            name=base.name, revision=base.revision, dtype=base.dtype,
            adapters=(_organism_adapter(cfg),),
        )
    if arm.kind == "model_swap":
        if "ft_artifact" not in cfg:
            raise KeyError(
                f"arm {arm.name!r} is a model_swap: cfg must carry "
                "'ft_artifact' (the cell's saved realignment adapter dir "
                "for ft_lora_*) — the driver injects it per cell"
            )
        artifact = str(cfg["ft_artifact"])
        return ModelSpec(
            name=base.name, revision=base.revision, dtype=base.dtype,
            adapters=(_organism_adapter(cfg), (artifact, None)),
        )
    raise ValueError(f"unknown arm kind {arm.kind!r} for arm {arm.name!r}")
