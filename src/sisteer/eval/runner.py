"""Experiment orchestration: fit vectors, sweep layers/scales, evaluate arms.

The unit of work is (model, instruction, method). Outputs land in
<out_dir>/<instruction>/<method>/ as:
  vector.safetensors + vector.json   (with extraction diagnostics)
  results.json                       (all arms, per-prompt records)
and are mirrored to W&B when enabled.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..data.instructions import Instruction
from ..data.pairs import (
    Pair,
    generate_pairs,
    instruction_diff_inputs,
    pairs_to_caa_inputs,
    pairs_to_distill_examples,
    pairs_to_pref_examples,
    save_pairs,
)
from ..extraction.actadd import extract_actadd
from ..extraction.bidpo import train_bidpo
from ..extraction.diffmean import extract_caa, extract_instruction_diff
from ..extraction.distill import train_distill
from ..models import num_layers
from ..vectors import SteeringVector
from .behavioral import anti_steerable_fraction, eval_instruction
from .capability import distinct_ngram_ratio, mmlu_accuracy, perplexity

METHODS = ("actadd", "caa", "instr_diff", "bidpo", "distill_soft", "distill_hard")


def default_layers(model) -> list[int]:
    """Sweep band: ~30-75% of depth (CAA layer 13/32; Stolfo mostly mid;
    ActAdd abstract concepts late-mid)."""
    n = num_layers(model)
    lo, hi = int(0.3 * n), int(0.75 * n)
    step = max(1, (hi - lo) // 6)
    return list(range(lo, hi + 1, step))


def make_wandb_logger(project: str, run_name: str, config: dict):
    if os.environ.get("WANDB_MODE") == "disabled":
        return None, None
    from sisteer import site as _site

    # $WANDB_PROJECT (a config's placeholder, or the default below) -> WANDB_PROJECT / SISTEER_WANDB_PROJECT; one with no
    # value on this host turns W&B off for the run, said on stderr with the variable to set (W&B is optional here)
    try:
        project = _site.expand(project, key="wandb_project")
    except _site.PlaceholderError as exc:
        print(f"[wandb] disabled: {exc}", file=sys.stderr)
        return None, None
    import wandb

    run = wandb.init(project=project, name=run_name, config=config, reinit=True)

    def log(metrics: dict, step: int | None = None):
        wandb.log(metrics, step=step)

    return run, log


def fit_vector(
    method: str,
    model,
    tok,
    instruction: Instruction,
    pairs: list[Pair],
    fit_prompts: list[str],
    layer: int,
    *,
    train_kwargs: dict | None = None,
    log_fn=None,
) -> SteeringVector:
    train_kwargs = train_kwargs or {}
    if method == "actadd":
        return extract_actadd(
            model, tok,
            prompt_plus=instruction.text,
            prompt_minus=instruction.anti_text or "",
            paper_layers=[max(layer, 1)],
            instruction_id=instruction.id,
            pool=train_kwargs.get("pool", True),
        )
    if method == "caa":
        prompts, pos, neg = pairs_to_caa_inputs(tok, pairs)
        return extract_caa(
            model, tok, prompts, pos, neg, layers=[layer],
            position=train_kwargs.get("position", "completion_mean"),
            instruction_id=instruction.id,
        )
    if method == "instr_diff":
        with_si, without = instruction_diff_inputs(tok, instruction, fit_prompts)
        return extract_instruction_diff(
            model, tok, with_si, without, layers=[layer],
            position=train_kwargs.get("position", "last"),
            instruction_id=instruction.id,
        )
    if method == "bidpo":
        examples = pairs_to_pref_examples(tok, pairs)
        return train_bidpo(
            model, tok, examples, layer,
            instruction_id=instruction.id, log_fn=log_fn, **train_kwargs,
        )
    if method in ("distill_soft", "distill_hard"):
        examples = pairs_to_distill_examples(tok, pairs, instruction)
        return train_distill(
            model, tok, examples, layer,
            loss=method.removeprefix("distill_"),
            instruction_id=instruction.id, log_fn=log_fn, **train_kwargs,
        )
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


def sweep_layers(
    method: str,
    model,
    tok,
    instruction: Instruction,
    pairs: list[Pair],
    fit_prompts: list[str],
    val_prompts: list[str],
    layers: list[int],
    scale: float = 1.0,
    rep_threshold: float = 0.35,
    ppl_guard_factor: float = 10.0,
    ppl_texts: list[str] | None = None,
    **fit_kw,
) -> tuple[int, SteeringVector, dict]:
    """Stolfo-style selection: fit per layer, score strict compliance on
    validation prompts at a reference scale, discard degenerate layers,
    pick the best survivor; ties -> earliest layer.

    Degeneration is judged primarily by n-gram repetition of the steered
    validation generations (neutral-corpus perplexity is confounded for
    instructions that legitimately shift the output distribution — a
    French vector raises English NLL while producing fine text); a loose
    10x perplexity backstop still catches total collapse."""
    base_ppl = perplexity(model, tok, texts=ppl_texts, n_texts=10)
    best: tuple[float, int, SteeringVector] | None = None
    sweep_log = {}
    for layer in layers:
        sv = fit_vector(
            method, model, tok, instruction, pairs, fit_prompts, layer, **fit_kw
        )
        arms = eval_instruction(
            model, tok, instruction, val_prompts, sv, scales=[scale]
        )
        steered = [a for a in arms if a.arm == "steered"][0]
        responses = [r["response"] for r in steered.records]
        rep = distinct_ngram_ratio(responses)
        ppl = perplexity(model, tok, texts=ppl_texts, sv=sv, scale=scale, n_texts=10)
        degenerate = rep < rep_threshold or ppl > ppl_guard_factor * base_ppl
        sweep_log[layer] = {
            "compliance": steered.compliance_strict,
            "distinct3": rep,
            "ppl": ppl,
            "degenerate": degenerate,
        }
        if degenerate:
            continue
        score = steered.compliance_strict
        if best is None or score > best[0]:
            best = (score, layer, sv)
    if best is None:  # everything degenerate: fall back to first layer
        layer = layers[0]
        sv = fit_vector(
            method, model, tok, instruction, pairs, fit_prompts, layer, **fit_kw
        )
        best = (0.0, layer, sv)
    _, layer, sv = best
    return layer, sv, sweep_log


def run_instruction_method(
    model,
    tok,
    instruction: Instruction,
    method: str,
    *,
    fit_prompts: list[str],
    eval_prompts: list[str],
    nontrigger_prompts: list[str] | None = None,
    layers: list[int] | None = None,
    scales: list[float] | None = None,
    out_dir: str | Path = "out",
    pairs: list[Pair] | None = None,
    train_kwargs: dict | None = None,
    capability: bool = True,
    mmlu_rows: list[dict] | None = None,
    ppl_texts: list[str] | None = None,
    wandb_project: str | None = "$WANDB_PROJECT",
    max_new_tokens: int = 200,
) -> dict:
    layers = layers or default_layers(model)
    scales = scales if scales is not None else [0.5, 1.0, 2.0, 4.0]
    out = Path(out_dir) / instruction.id / method
    out.mkdir(parents=True, exist_ok=True)

    run = log_fn = None
    if wandb_project:
        run, log_fn = make_wandb_logger(
            wandb_project,
            f"{instruction.id}-{method}",
            {"instruction": instruction.id, "method": method, "layers": layers},
        )

    needs_pairs = method in ("caa", "bidpo", "distill_soft", "distill_hard")
    if needs_pairs and pairs is None:
        pairs = generate_pairs(
            model, tok, instruction, fit_prompts, max_new_tokens=max_new_tokens
        )
        save_pairs(pairs, out / "pairs.jsonl")
    pairs = pairs or []

    # Validation subset (from fit prompts, disjoint from eval) for the sweep.
    val_prompts = fit_prompts[: min(8, len(fit_prompts))]
    layer, sv, sweep_log = sweep_layers(
        method, model, tok, instruction, pairs, fit_prompts, val_prompts,
        layers, ppl_texts=ppl_texts, train_kwargs=train_kwargs, log_fn=log_fn,
    )
    sv.metadata["selected_layer"] = layer
    sv.metadata["layer_sweep"] = sweep_log
    sv.save(out / "vector.safetensors")

    arms = eval_instruction(
        model, tok, instruction, eval_prompts, sv, scales,
        nontrigger_prompts=nontrigger_prompts,
        include_stacking=True,
        max_new_tokens=max_new_tokens,
    )
    baseline = [a for a in arms if a.arm == "baseline"][0]
    summary = {
        "instruction": instruction.id,
        "category": instruction.category,
        "method": method,
        "layer": layer,
        "arms": [a.summary() for a in arms],
        "anti_steerable": {
            str(a.scale): anti_steerable_fraction(a, baseline)
            for a in arms
            if a.arm == "steered"
        },
    }
    if capability:
        steered_arms = [a for a in arms if a.arm == "steered"]
        best_scale = max(
            steered_arms, key=lambda a: a.compliance_strict
        ).scale if steered_arms else scales[-1]
        cap = {"unsteered": {
            "mmlu": mmlu_accuracy(model, tok, rows=mmlu_rows),
            "ppl": perplexity(model, tok, texts=ppl_texts),
        }}
        for s in sorted({best_scale, scales[-1]}):
            cap[f"scale_{s}"] = {
                "mmlu": mmlu_accuracy(model, tok, sv=sv, scale=s, rows=mmlu_rows),
                "ppl": perplexity(model, tok, texts=ppl_texts, sv=sv, scale=s),
            }
        cap["best_scale"] = best_scale
        summary["capability"] = cap

    (out / "results.json").write_text(
        json.dumps(
            {**summary, "records": [a.records for a in arms]}, indent=1
        )
    )
    if run is not None:
        import wandb

        table = wandb.Table(
            columns=["arm", "scale", "strict", "loose", "false_fire"],
            data=[
                [a.arm, a.scale, a.compliance_strict, a.compliance_loose,
                 a.false_fire_strict]
                for a in arms
            ],
        )
        run.log({"arms": table, **{
            f"compliance/{a.arm}@{a.scale}": a.compliance_strict for a in arms
        }})
        run.finish()
    return summary
