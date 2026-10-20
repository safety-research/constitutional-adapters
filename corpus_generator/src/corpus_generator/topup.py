"""Top-up sizing: compute the binding deficit per axis, size extension waves
at ~90% LOWER confidence bound yields, and write TOPUP-NNN.json specs (the
next `corpusgen run` executes the extension sub-DAGs; base artifacts untouched)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import RunConfig
from .io import atomic_write
from .pipeline import dag_expanded_totals, load_topups
from .tokens import official_tokens, record_min_tokens

Z90 = 1.2816  # one-sided ~90% normal quantile


def plan_topup(
    cfg: RunConfig,
    run_dir: Path,
    corpus: str,
    finals: list[dict],
    stories_n: int | None = None,
    tokens_n: int | None = None,
) -> dict | None:
    """Compute the binding deficits for a corpus (target minus achieved, or
    the explicit overrides) and write the sized extension spec, if any."""
    official = official_tokens(finals)
    if corpus == "advice":
        deficit = tokens_n if tokens_n is not None else cfg.targets.advice_min_tokens - official
        return plan_advice_topup(
            cfg, run_dir, finals, deficit,
            attempted_scenarios=dag_expanded_totals(cfg)["advice_scenarios"],
        )
    count_deficit = (
        stories_n if stories_n is not None else cfg.targets.stories_min_count - len(finals)
    )
    tok_deficit = tokens_n if tokens_n is not None else cfg.targets.stories_min_tokens - official
    return plan_stories_topup(
        cfg, run_dir, finals, count_deficit, tok_deficit,
        attempted_calls=cfg.stories.n_calls,
    )


def _lcb_rate(successes: int, trials: int) -> float:
    if trials == 0:
        return 0.0
    p = successes / trials
    lcb = p - Z90 * math.sqrt(max(p * (1 - p), 1e-9) / trials)
    return max(lcb, 1e-3)


def _lcb_mean(values: list[float]) -> float:
    if not values:
        return 1.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean * 0.8
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return max(mean - Z90 * math.sqrt(var / n), mean * 0.25)


def _next_ordinal(run_dir: Path, corpus: str) -> int:
    return len(load_topups(run_dir, corpus)) + 1


def plan_advice_topup(
    cfg: RunConfig,
    run_dir: Path,
    final_records: list[dict],
    tokens_deficit: int,
    attempted_scenarios: int,
) -> dict | None:
    """Extension = new scenario slots on EVERY (principle, type) pair."""
    if tokens_deficit <= 0:
        return None
    n_final = len(final_records)
    per_rec = [t for t in map(record_min_tokens, final_records) if t is not None]
    yield_lcb = _lcb_rate(n_final, max(attempted_scenarios, 1))
    tok_lcb = _lcb_mean(per_rec) if per_rec else 800.0
    extra_scenarios = math.ceil(tokens_deficit / (yield_lcb * tok_lcb))

    # prompt_type_id is already the pair key "pNN-tNN" (A2 unit key)
    pairs = sorted({r["prompt_type_id"] for r in final_records}) or [
        f"p{i:02d}-t{j:02d}"
        for i in range(cfg.advice.n_principles)
        for j in range(cfg.advice.n_types)
    ]
    n_slots = math.ceil(extra_scenarios / len(pairs))
    n = _next_ordinal(run_dir, "advice")
    prior_slots = sum(t["n_slots"] for t in load_topups(run_dir, "advice"))
    spec = {
        "n": n,
        "pairs": pairs,
        "slot_start": cfg.advice.n_scenarios_per_call + prior_slots,
        "n_slots": n_slots,
    }
    atomic_write(
        run_dir / "advice" / f"TOPUP-{n:03d}.json", json.dumps(spec, indent=1)
    )
    return spec


def plan_stories_topup(
    cfg: RunConfig,
    run_dir: Path,
    final_records: list[dict],
    stories_deficit: int,
    tokens_deficit: int,
    attempted_calls: int,
) -> dict | None:
    """Extension = new call indices on every (category, subcategory) lane.
    Only the BINDING axis drives the size; both are checked independently."""
    if stories_deficit <= 0 and tokens_deficit <= 0:
        return None
    n_final = len(final_records)
    per_rec = [t for t in map(record_min_tokens, final_records) if t is not None]
    per_call_yield = _lcb_rate(n_final, max(attempted_calls * cfg.stories.stories_per_call, 1))
    stories_per_call_lcb = per_call_yield * cfg.stories.stories_per_call
    tok_lcb = _lcb_mean(per_rec) if per_rec else 1500.0

    calls_for_count = (
        math.ceil(stories_deficit / stories_per_call_lcb) if stories_deficit > 0 else 0
    )
    calls_for_tokens = (
        math.ceil(tokens_deficit / (stories_per_call_lcb * tok_lcb))
        if tokens_deficit > 0
        else 0
    )
    extra_calls = max(calls_for_count, calls_for_tokens)
    calls_per_lane = max(1, math.ceil(extra_calls / cfg.stories.n_lanes))
    n = _next_ordinal(run_dir, "stories")
    prior_calls = sum(t["n_calls"] for t in load_topups(run_dir, "stories"))
    spec = {
        "n": n,
        "call_start": cfg.stories.calls_per_subcategory + prior_calls,
        "n_calls": calls_per_lane,
    }
    atomic_write(
        run_dir / "stories" / f"TOPUP-{n:03d}.json", json.dumps(spec, indent=1)
    )
    return spec
