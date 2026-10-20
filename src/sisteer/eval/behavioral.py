"""On-target behavioral evaluation: compliance rates under four arms.

Arms per instruction (the core comparison of the project):
  baseline   — no SI, no vector
  prompted   — SI in context (the thing the vector must match)
  steered    — vector, no SI, swept over a scale grid
  both       — SI in context + vector (stacking, CAA Table 3 style)

For conditional instructions, compliance is a 2x2: trigger prompts must
fire, non-trigger prompts must NOT (false-fire = CAST's indiscriminate-
steering failure). Per-prompt records are kept for every scale so
dose-response slopes and anti-steerable fractions (Tan et al.) can be
computed offline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..data.instructions import Instruction
from ..generate import generate_batch
from ..models import build_chat_prompt
from ..vectors import SteeringVector
from . import checkers


@dataclass
class ArmResult:
    arm: str  # baseline | prompted | steered | both
    scale: float
    compliance_strict: float
    compliance_loose: float
    false_fire_strict: float | None = None  # conditional instructions only
    records: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("records")
        return d


def _score(
    instruction: Instruction, prompts: list[str], responses: list[str]
) -> tuple[float, float, list[dict]]:
    records = []
    n_strict = n_loose = 0
    for p, r in zip(prompts, responses):
        s = checkers.check(instruction.checker, r, **instruction.checker_kwargs)
        l = checkers.check_loose(instruction.checker, r, **instruction.checker_kwargs)
        n_strict += s
        n_loose += l
        records.append({"prompt": p, "response": r, "strict": s, "loose": l})
    n = max(len(prompts), 1)
    return n_strict / n, n_loose / n, records


def eval_arm(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    *,
    arm: str,
    sv: SteeringVector | None = None,
    scale: float = 0.0,
    positions: str = "all",
    norm_relative: bool = False,
    project_targets: dict[int, float] | None = None,
    nontrigger_prompts: list[str] | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 200,
) -> ArmResult:
    system = instruction.text if arm in ("prompted", "both") else None
    use_sv = sv if arm in ("steered", "both") else None

    rendered = [build_chat_prompt(tok, p, system=system) for p in prompts]
    responses = generate_batch(
        model, tok, rendered,
        sv=use_sv, scale=scale, positions=positions,
        norm_relative=norm_relative, project_targets=project_targets,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    strict, loose, records = _score(instruction, prompts, responses)

    false_fire = None
    if nontrigger_prompts:
        nt_rendered = [
            build_chat_prompt(tok, p, system=system) for p in nontrigger_prompts
        ]
        nt_responses = generate_batch(
            model, tok, nt_rendered,
            sv=use_sv, scale=scale, positions=positions,
            norm_relative=norm_relative, project_targets=project_targets,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        ff_strict, _, nt_records = _score(
            instruction, nontrigger_prompts, nt_responses
        )
        false_fire = ff_strict  # checker passing off-trigger = false fire
        for rec in nt_records:
            rec["nontrigger"] = True
        records.extend(nt_records)

    return ArmResult(
        arm=arm,
        scale=scale if use_sv is not None else 0.0,
        compliance_strict=strict,
        compliance_loose=loose,
        false_fire_strict=false_fire,
        records=records,
    )


def eval_instruction(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    sv: SteeringVector | None,
    scales: list[float],
    *,
    positions: str = "all",
    norm_relative: bool = False,
    project_targets: dict[int, float] | None = None,
    nontrigger_prompts: list[str] | None = None,
    include_stacking: bool = False,
    batch_size: int = 8,
    max_new_tokens: int = 200,
) -> list[ArmResult]:
    common = dict(
        positions=positions,
        norm_relative=norm_relative,
        project_targets=project_targets,
        nontrigger_prompts=nontrigger_prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    results = [
        eval_arm(model, tok, instruction, prompts, arm="baseline", **common),
        eval_arm(model, tok, instruction, prompts, arm="prompted", **common),
    ]
    if sv is not None:
        for s in scales:
            results.append(
                eval_arm(
                    model, tok, instruction, prompts,
                    arm="steered", sv=sv, scale=s, **common,
                )
            )
        if include_stacking:
            results.append(
                eval_arm(
                    model, tok, instruction, prompts,
                    arm="both", sv=sv, scale=scales[len(scales) // 2], **common,
                )
            )
    return results


def eval_arm_obj(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    *,
    arm: str,
    obj=None,
    scale: float = 0.0,
    batch_size: int = 8,
    max_new_tokens: int = 200,
) -> ArmResult:
    """Object-generic twin of eval_arm: the intervention is ANY object with
    apply(model, scale)/inject(prompt) (the stage-2 arm protocol), not a
    SteeringVector. Token sentinels and per-chunk apply() are handled by
    generate_with_object."""
    from ..generate import generate_with_object

    system = instruction.text if arm in ("prompted", "both") else None
    use_obj = obj if arm in ("steered", "both") else None
    rendered = [build_chat_prompt(tok, p, system=system) for p in prompts]
    responses = generate_with_object(
        model, tok, rendered, obj=use_obj, scale=scale,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    strict, loose, records = _score(instruction, prompts, responses)
    return ArmResult(
        arm=arm,
        scale=scale if use_obj is not None else 0.0,
        compliance_strict=strict,
        compliance_loose=loose,
        records=records,
    )


def eval_instruction_obj(
    model,
    tok,
    instruction: Instruction,
    prompts: list[str],
    obj,
    scales: list[float],
    *,
    include_stacking: bool = True,
    stack_scale: float | None = None,
    batch_size: int = 8,
    max_new_tokens: int = 200,
) -> list[ArmResult]:
    """baseline / prompted / steered@each scale / both, for any arm object
    (the stage-2 matrix cell's arm loop). stack_scale pins the "both"
    (stacking) arm's scale explicitly; None keeps the historical default,
    the grid's median point. Callers whose grid can GROW per-arm
    (extra_scales) pass the BASE grid's median so an added factor never
    silently moves the stacking point out from under the existing matrix."""
    common = dict(batch_size=batch_size, max_new_tokens=max_new_tokens)
    results = [
        eval_arm_obj(model, tok, instruction, prompts, arm="baseline", **common),
        eval_arm_obj(model, tok, instruction, prompts, arm="prompted", **common),
    ]
    if obj is not None:
        for s in scales:
            results.append(
                eval_arm_obj(model, tok, instruction, prompts,
                             arm="steered", obj=obj, scale=s, **common)
            )
        if include_stacking:
            if stack_scale is None:
                stack_scale = scales[len(scales) // 2]
            results.append(
                eval_arm_obj(model, tok, instruction, prompts, arm="both",
                             obj=obj, scale=stack_scale, **common)
            )
    return results


def anti_steerable_fraction(
    steered: ArmResult, baseline: ArmResult
) -> float | None:
    """Fraction of prompts that were compliant unsteered but broke under
    steering (per-input regression, Tan/Braun-style but for binary
    compliance)."""
    base = {r["prompt"]: r["strict"] for r in baseline.records if not r.get("nontrigger")}
    flips = total = 0
    for r in steered.records:
        if r.get("nontrigger") or r["prompt"] not in base:
            continue
        total += 1
        if base[r["prompt"]] and not r["strict"]:
            flips += 1
    return flips / total if total else None
