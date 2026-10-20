"""Arm runner for the agentic-misalignment held-out suite.

Renders each scenario with the safety SI composed FIRST, then the scenario's
own scaffold system prompt (composition order pre-registered as
safety-SI-first), generates a free-form response under the arm's intervention
(prompt / object / model-swap), and appends one record per scenario. It
NEVER scores — free-form output goes to the Sonnet judge only.

Records carry the full F13 contract: ``item_sha`` (arm-independent, for
pairing), ``prompt_sha`` (the complete rendered prompt, for
intervention/resume verification), and ``suite_revision``.
"""

from __future__ import annotations

from ...generate import generate_with_object
from ...models import build_chat_prompt
from ..realign_suites import (
    append_records,
    arm_name,
    compose_system,
    item_sha,
    prompt_sha,
    read_done_ids,
)
from .loader import SUITE_REVISION


def run_agentic_arm(
    model,
    tok,
    scenarios: list[dict],
    *,
    system: str | None = None,
    obj=None,
    scale: float = 1.0,
    out_jsonl,
    max_new_tokens: int = 600,
    batch_size: int = 8,
) -> list[dict]:
    """Generate + record the arm's response for each scenario.

    ``system`` is the safety SI text (the prompting arm's intervention, or a
    prompt component stacked with ``obj``); ``obj`` is any intervention object
    (``None`` for baseline / a swapped model). Item-level resume: scenarios
    whose ``scenario_id`` is already in ``out_jsonl`` are skipped.
    """
    arm = arm_name(system, obj)
    done = read_done_ids(out_jsonl, "scenario_id")
    pending = [s for s in scenarios if s["scenario_id"] not in done]

    written: list[dict] = []
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        rendered = [
            build_chat_prompt(
                tok,
                s["user_prompt"],
                system=compose_system(system, s.get("system_prompt")),
            )
            for s in chunk
        ]
        responses = generate_with_object(
            model, tok, rendered, obj=obj, scale=scale,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        records = []
        for s, prompt, response in zip(chunk, rendered, responses):
            records.append(
                {
                    "scenario_id": s["scenario_id"],
                    "behavior": s["behavior"],
                    "condition": s["condition"],
                    "response": response,
                    "arm": arm,
                    "suite_revision": SUITE_REVISION,
                    "item_sha": item_sha(
                        {
                            "scenario_id": s["scenario_id"],
                            "behavior": s["behavior"],
                            "condition": s["condition"],
                            "system_prompt": s.get("system_prompt"),
                            "user_prompt": s["user_prompt"],
                        }
                    ),
                    "prompt_sha": prompt_sha(prompt),
                }
            )
        append_records(out_jsonl, records)
        written.extend(records)
    return written
