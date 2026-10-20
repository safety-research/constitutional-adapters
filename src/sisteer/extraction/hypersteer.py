"""Minimal No-Context HyperSteer (Sun et al. 2506.03292).

Amortized instruction->vector map: instead of one trained vector per
instruction, a small head maps a frozen embedding of the instruction
text to the steering vector, trained jointly across many instructions
with the same steered-LM loss as distill.py (their Eq. 4 is the CE
case).

Simplifications vs the paper, per its own ablations:
- encoder = the frozen TARGET model itself (mean-pooled residual state of
  the instruction text at a mid layer), not a trainable Gemma copy —
  their depth ablation shows a tiny head loses little held-in;
- free-norm output head (like their cross-attention variant), no
  unit-norm + detection-latent magnitude machinery;
- strength is swept at eval time on a scalar grid regardless.

At our scale (hundreds of instructions, not 16k) expect held-in
"amortized fitting" behavior, not held-out generalization.
"""

from __future__ import annotations

import torch
from torch import nn

from ..models import hidden_size
from ..vectors import SteeringVector
from .capture import collect_pooled_activations
from .distill import DistillExample, train_steered_lm


class HyperNetNC(nn.Module):
    """instruction embedding [K, d] -> steering vector [d]."""

    def __init__(self, instr_reps: torch.Tensor, d_model: int, hidden: int = 1024):
        super().__init__()
        self.register_buffer("instr_reps", instr_reps)
        self.mlp = nn.Sequential(
            nn.Linear(instr_reps.shape[1], hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # Near-zero init so training starts from the unsteered model,
        # matching the zero-init convention of the fixed-vector methods.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, idx: int) -> torch.Tensor:
        return self.mlp(self.instr_reps[idx])


@torch.no_grad()
def embed_instructions(
    model, tok, instruction_texts: list[str], embed_layer: int
) -> torch.Tensor:
    """Frozen mean-pooled residual embedding of each instruction text."""
    acts = collect_pooled_activations(
        model, tok, instruction_texts, [embed_layer], position="mean"
    )
    return acts.acts[embed_layer]


def train_hypersteer_nc(
    model,
    tok,
    examples: list[DistillExample],
    instruction_texts: list[str],
    layer: int,
    *,
    embed_layer: int | None = None,
    hidden: int = 1024,
    loss: str = "hard",
    lr: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 4,
    max_len: int = 768,
    positions: str = "all",
    skip_bos: bool = True,
    seed: int = 0,
    log_fn=None,
) -> tuple[HyperNetNC, list[SteeringVector]]:
    """examples[i].instruction_idx indexes into instruction_texts.

    Returns the trained hypernet plus one materialized SteeringVector per
    training instruction (so downstream eval treats them like any other
    method's vectors)."""
    device = next(model.parameters()).device
    if embed_layer is None:
        from ..models import num_layers

        embed_layer = num_layers(model) // 2
    reps = embed_instructions(model, tok, instruction_texts, embed_layer).to(device)
    hyper = HyperNetNC(reps, hidden_size(model), hidden=hidden).to(device)
    hyper = hyper.float()

    history = train_steered_lm(
        model,
        tok,
        examples,
        layer,
        vector_provider=lambda idxs: hyper(idxs[0]),
        trainable_params=list(hyper.mlp.parameters()),
        loss=loss,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        positions=positions,
        skip_bos=skip_bos,
        seed=seed,
        log_fn=log_fn,
    )

    vectors = []
    with torch.no_grad():
        for k, text in enumerate(instruction_texts):
            vec = hyper(k).detach().float().cpu()
            vectors.append(
                SteeringVector(
                    vectors={layer: vec},
                    method="hypersteer_nc",
                    model_name=model.config._name_or_path,
                    instruction_id=f"idx_{k}",
                    metadata={
                        "instruction_text": text,
                        "embed_layer": embed_layer,
                        "loss": loss,
                        "final_loss": history[-1] if history else None,
                    },
                )
            )
    return hyper, vectors
