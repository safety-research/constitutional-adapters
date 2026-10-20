"""Steering-vector container with save/load (safetensors + JSON sidecar)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


@dataclass
class SteeringVector:
    """A per-layer additive residual-stream intervention.

    vectors maps layer index -> tensor. Shape [d_model] for a single
    direction, or [P, d_model] for per-position vectors (ActAdd-style
    front injection).
    """

    vectors: dict[int, torch.Tensor]
    method: str
    model_name: str
    instruction_id: str
    metadata: dict = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        return sorted(self.vectors)

    def scaled(self, factor: float) -> "SteeringVector":
        return SteeringVector(
            vectors={k: v * factor for k, v in self.vectors.items()},
            method=self.method,
            model_name=self.model_name,
            instruction_id=self.instruction_id,
            metadata={**self.metadata, "scaled_by": factor},
        )

    def norms(self) -> dict[int, float]:
        return {k: float(v.norm()) for k, v in self.vectors.items()}

    def to(self, device=None, dtype=None) -> "SteeringVector":
        return SteeringVector(
            vectors={
                k: v.to(device=device, dtype=dtype) for k, v in self.vectors.items()
            },
            method=self.method,
            model_name=self.model_name,
            instruction_id=self.instruction_id,
            metadata=self.metadata,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {f"layer_{k}": v.contiguous().cpu() for k, v in self.vectors.items()}
        save_file(tensors, str(path))
        sidecar = {
            "method": self.method,
            "model_name": self.model_name,
            "instruction_id": self.instruction_id,
            "layers": self.layers,
            "norms": self.norms(),
            "metadata": self.metadata,
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SteeringVector":
        path = Path(path)
        tensors = load_file(str(path))
        # COMPATIBILITY SHIM, not the fix. The PRIMARY fix is upstream on the
        # SAVE side: train_two_si_objects._save_object (commit ac7d464,
        # accel-fit) now writes a correctly-named sidecar and its publish step
        # moves every file the save produced to its canonical name before
        # publishing the data file, so NEW artifacts arrive as <stem>.json.
        # This loader-side tolerance stays anyway because the stage-3 wave's
        # ALREADY-FIT artifacts carry the OLD name, cannot be re-fit cheaply,
        # and would otherwise be unreadable — a shim for existing data, not a
        # substitute for the save fix.
        #
        # WHO IS AFFECTED (the durable predicate): ANY save routed through a
        # TEMP FILENAME, not "the vector family". SteeringVector.save writes the
        # sidecar at ``save_path.with_suffix('.json')``; a caller that saves to
        # ``object.safetensors.tmp`` then atomically renames only the DATA file
        # to ``object.safetensors`` leaves the sidecar at
        # ``object.safetensors.json``, because with_suffix strips only the LAST
        # suffix (``.tmp``), not ``.safetensors``. A DIRECT save (no temp) —
        # fit_class_b's antidote — gets the canonical ``object.json`` and needs
        # no fallback; naming the culprit "the vector family" is exactly what
        # stopped anyone explaining why antidote was correct.
        #
        # So: prefer the canonical <stem>.json; fall back to <fullname>.json
        # only when the canonical is absent. CANONICAL WINS when both exist — a
        # tree now really does contain both (a mix of upstream-fixed and
        # already-fit artifacts), so a stale misplaced sidecar must never shadow
        # a correct one.
        sidecar_path = path.with_suffix(".json")
        if not sidecar_path.exists():
            alt = path.with_name(path.name + ".json")
            if alt.exists():
                sidecar_path = alt
        sidecar = json.loads(sidecar_path.read_text())
        vectors = {
            int(name.removeprefix("layer_")): t for name, t in tensors.items()
        }
        return cls(
            vectors=vectors,
            method=sidecar["method"],
            model_name=sidecar["model_name"],
            instruction_id=sidecar["instruction_id"],
            metadata=sidecar.get("metadata", {}),
        )
