"""ActAdd (Turner et al. 2308.10248): single contrast-pair activation diff.

Faithful details (verified against the paper v5 + montemac/activation_additions):
- activations are read at hook_resid_pre of paper-layer l — the residual
  stream ENTERING block l. Our harness convention indexes block OUTPUTS,
  and resid_pre(l) == resid_post(l-1), so the vector is stored under
  harness layer l-1 (paper_layer >= 1 required).
- the pair is padded to token parity in TOKEN space using the literal
  space token (not the tokenizer pad/EOS — EOT "works notably poorly").
- the steering object is the full per-position tensor [P, d_model],
  injected at the FRONT positions of the prompt during prefill only
  (hooks positions="front" reproduces this: decode steps have offsets
  >= P, so nothing is added after prefill).
- no normalization; working coefficients are large (c in [3, 20]).

A mean-pooled single-vector form (pool=True) is exposed so ActAdd can be
compared in the same slot as the other methods; report it separately —
it is a deviation from the published method.
"""

from __future__ import annotations

import torch

from ..vectors import SteeringVector
from .capture import capture_layer_outputs


@torch.no_grad()
def extract_actadd(
    model,
    tok,
    prompt_plus: str,
    prompt_minus: str,
    paper_layers: list[int],
    instruction_id: str = "",
    pool: bool = False,
) -> SteeringVector:
    if any(l < 1 for l in paper_layers):
        raise ValueError(
            "paper_layer 0 reads the embedding output, which is not reachable "
            "as a block output in this harness; use paper_layers >= 1"
        )
    device = next(model.parameters()).device
    ids_p = tok(prompt_plus)["input_ids"]
    ids_m = tok(prompt_minus)["input_ids"]
    space_id = tok(" ", add_special_tokens=False)["input_ids"]
    if len(space_id) != 1:
        space_id = [tok.pad_token_id]
    while len(ids_p) < len(ids_m):
        ids_p = ids_p + space_id
    while len(ids_m) < len(ids_p):
        ids_m = ids_m + space_id

    def run(ids):
        t = torch.tensor([ids], device=device)
        with capture_layer_outputs(model, paper_layers, site="pre") as captured:
            model(input_ids=t)
        return {l: captured[l][0].float().cpu() for l in paper_layers}  # [P, d]

    acts_p = run(ids_p)
    acts_m = run(ids_m)
    vectors = {}
    for l in paper_layers:
        diff = acts_p[l] - acts_m[l]  # [P, d]
        vectors[l - 1] = diff.mean(0) if pool else diff
    return SteeringVector(
        vectors=vectors,
        method="actadd" + ("_pooled" if pool else ""),
        model_name=model.config._name_or_path,
        instruction_id=instruction_id,
        metadata={
            "site": "resid_pre",
            "paper_layers": paper_layers,
            "prompt_plus": prompt_plus,
            "prompt_minus": prompt_minus,
            "n_positions": len(ids_p),
            "pooled": pool,
        },
    )
