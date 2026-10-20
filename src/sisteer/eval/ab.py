"""A/B (multiple-choice) propensity evaluation — CAA's primary metric,
plus Tan et al.'s per-input dose-response machinery.

p(matching) = probability of the behavior-matching answer letter,
normalized over {A, B}, read from the next-token logits after a
pre-filled "(" (CAA's get_a_b_probs). Steering is applied to the
positions after the user prompt (which here is just the "(", matching
CAA's post-[/INST] injection for this format).
"""

from __future__ import annotations

import torch

from ..data.mwe import ABItem
from ..hooks import steering
from ..models import build_chat_prompt
from ..vectors import SteeringVector


def render_ab_prompt(tok, question: str, system: str | None = None,
                     base_format: bool = False) -> str:
    if base_format:  # CAA's base-model format
        prefix = f"{system.strip()}\n\n" if system else ""
        return f"{prefix}Input: {question}\nResponse: ("
    return build_chat_prompt(tok, question, system=system) + "("


def _letter_ids(tok) -> tuple[int, int]:
    ida = tok("(A", add_special_tokens=False)["input_ids"][-1]
    idb = tok("(B", add_special_tokens=False)["input_ids"][-1]
    return ida, idb


@torch.no_grad()
def ab_propensity(
    model,
    tok,
    items: list[ABItem],
    sv: SteeringVector | None = None,
    scale: float = 1.0,
    positions: str = "response",
    system: str | None = None,
    base_format: bool = False,
    batch_size: int = 8,
) -> list[float]:
    """Returns per-item p(matching)."""
    device = next(model.parameters()).device
    ida, idb = _letter_ids(tok)
    prev_side = tok.padding_side
    tok.padding_side = "left"
    probs: list[float] = []
    try:
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            texts = [
                render_ab_prompt(tok, it.question, system, base_format)
                for it in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            width = enc["input_ids"].shape[1]
            with steering(
                model, sv, scale=scale, positions=positions,
                prompt_len=width - 1,  # steer the pre-filled "(" onward
            ) as handle:
                if handle is not None:
                    handle.reset(prompt_len=width - 1)
                logits = model(**enc).logits[:, -1]
            two = torch.stack([logits[:, ida], logits[:, idb]], dim=1)
            p = torch.softmax(two.float(), dim=1)
            for it, row in zip(chunk, p):
                probs.append(float(row[0] if it.matching_letter == "A" else row[1]))
    finally:
        tok.padding_side = prev_side
    return probs


def dose_response(
    model, tok, items, sv, multipliers: list[float], **kw
) -> dict[float, list[float]]:
    out = {}
    for m in multipliers:
        out[m] = ab_propensity(
            model, tok, items, sv=sv if m != 0 else None, scale=m, **kw
        )
    return out


def steerability(dose: dict[float, list[float]]) -> dict:
    """Tan et al.: per-item slope of p(matching) vs multiplier (least
    squares), dataset-level slope on the means, anti-steerable fraction
    (per-item slope < 0)."""
    mults = sorted(dose)
    if not dose[mults[0]]:
        raise ValueError("steerability() got an empty dose-response (no items)")
    x = torch.tensor(mults, dtype=torch.float64)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    n_items = len(dose[mults[0]])
    slopes = []
    for i in range(n_items):
        y = torch.tensor([dose[m][i] for m in mults], dtype=torch.float64)
        slopes.append(float((xc * (y - y.mean())).sum() / denom))
    means = torch.tensor(
        [sum(dose[m]) / n_items for m in mults], dtype=torch.float64
    )
    dataset_slope = float((xc * (means - means.mean())).sum() / denom)
    anti = sum(s < 0 for s in slopes) / n_items
    return {
        "per_item_slopes": slopes,
        "dataset_slope": dataset_slope,
        "anti_steerable_frac": anti,
        "mean_by_multiplier": {m: float(v) for m, v in zip(mults, means)},
    }
