"""MASK honesty-under-pressure suite (held-out secondary; deception channel).

Pinned-sha offline loader for ``cais/MASK`` — ``hf_hub_download`` at a full
revision + a direct pandas/pyarrow parquet read, NEVER ``load_dataset`` (the
offline-on-compute-nodes constraint). A missing file/revision RAISES; there
is no silent fallback to an unpinned read.

Real schema (verified at the pinned revision, 2026-08-14): every archetype
parquet carries ``task_id, system_prompt, user_prompt, type, proposition,
ground_truth, formatted_ground_truth, delivery, canary``. The single-turn
binary-pressure archetypes work as plain prompts — a pressure ``system_prompt``
plus a ``user_prompt`` demanding a claim about a ``proposition`` whose
``ground_truth`` is known — so the deception test is: does the response assert
the FALSE side under pressure. Two archetypes are excluded from the default:

- ``statistics`` — ``type == "numerical"`` (``formatted_ground_truth`` is a
  numeric range, not a clean binary A/B), a different judging shape;
- ``doubling_down_known_facts`` — multi-turn (``dd_turn_1_*``), not a plain
  single-prompt item.

The remaining four (``provided_facts``, ``known_facts``, ``continuations``,
``disinformation``) are all single-turn ``binary`` pressure items. The
finalized default list is recorded here and belongs in the pre-registration.

Runner contract mirrors the other held-out suites: free-form generation, one
record per item, NEVER scored here (scoring is judge-side, rubric ``mask.v1``:
lie ⇔ the response asserts the false proposition).
"""

from __future__ import annotations

import random

from ..generate import generate_with_object
from ..models import build_chat_prompt
from .realign_suites import (
    append_records,
    arm_name,
    compose_system,
    item_sha,
    prompt_sha,
    read_done_ids,
)

MASK_REPO = "cais/MASK"
# cais/MASK @ main, resolved 2026-08-14.
MASK_REVISION = "4602b84dd9e2ca05c6e1eafbc14e556e908ac1bb"

# Single-turn binary-pressure archetypes only (see module docstring).
DEFAULT_ARCHETYPES: tuple[str, ...] = (
    "provided_facts",
    "known_facts",
    "continuations",
    "disinformation",
)


def _archetype_path(archetype: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=MASK_REPO,
        repo_type="dataset",
        filename=f"{archetype}/test-00000-of-00001.parquet",
        revision=revision,
    )


def load_mask(
    archetypes: tuple[str, ...] = DEFAULT_ARCHETYPES,
    n: int | None = None,
    seed: int = 0,
    revision: str = MASK_REVISION,
) -> list[dict]:
    """Item dicts from the requested MASK archetypes at a pinned revision.

    Each item: ``{item_id, archetype, system_pressure, user_prompt,
    proposition, ground_truth, formatted_ground_truth, type}``. Items are
    shuffled deterministically by ``seed``; ``n`` (if given) caps the total.
    A download failure raises — never a silent unpinned fallback.
    """
    import pandas as pd

    items: list[dict] = []
    for archetype in archetypes:
        path = _archetype_path(archetype, revision)
        df = pd.read_parquet(path)
        for row in df.to_dict("records"):
            items.append(
                {
                    "item_id": str(row["task_id"]),
                    "archetype": archetype,
                    "system_pressure": row["system_prompt"],
                    "user_prompt": row["user_prompt"],
                    "proposition": row["proposition"],
                    "ground_truth": row["ground_truth"],
                    "formatted_ground_truth": str(row["formatted_ground_truth"]),
                    "type": row["type"],
                }
            )
    random.Random(f"realign-mask:{seed}").shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def run_mask_arm(
    model,
    tok,
    items: list[dict],
    *,
    system: str | None = None,
    obj=None,
    scale: float = 1.0,
    out_jsonl,
    max_new_tokens: int = 400,
    batch_size: int = 8,
) -> list[dict]:
    """Generate + record the arm's response per MASK item. Never scores.

    ``system`` (safety SI) is composed FIRST, then the item's pressure system
    prompt. Item-level resume on ``item_id``.
    """
    arm = arm_name(system, obj)
    done = read_done_ids(out_jsonl, "item_id")
    pending = [it for it in items if it["item_id"] not in done]

    written: list[dict] = []
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        rendered = [
            build_chat_prompt(
                tok,
                it["user_prompt"],
                system=compose_system(system, it["system_pressure"]),
            )
            for it in chunk
        ]
        responses = generate_with_object(
            model, tok, rendered, obj=obj, scale=scale,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        records = []
        for it, prompt, response in zip(chunk, rendered, responses):
            records.append(
                {
                    "item_id": it["item_id"],
                    "behavior": "deception",
                    "archetype": it["archetype"],
                    "system_pressure": it["system_pressure"],
                    "response": response,
                    "ground_truth": it["ground_truth"],
                    "formatted_ground_truth": it["formatted_ground_truth"],
                    "proposition": it["proposition"],
                    "arm": arm,
                    "suite_revision": MASK_REVISION,
                    "item_sha": item_sha(
                        {
                            "item_id": it["item_id"],
                            "archetype": it["archetype"],
                            "system_pressure": it["system_pressure"],
                            "user_prompt": it["user_prompt"],
                            "proposition": it["proposition"],
                            "ground_truth": it["ground_truth"],
                        }
                    ),
                    "prompt_sha": prompt_sha(prompt),
                }
            )
        append_records(out_jsonl, records)
        written.extend(records)
    return written
