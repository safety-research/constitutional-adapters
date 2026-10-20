"""``or_bench_hard_1k`` — an Inspect-shaped OR-Bench-Hard-1K.

Loaded by PATH from :mod:`sisteer.capeval.tiers.inspect_served`.

WHY THIS TASK EXISTS. The spec assumes inspect_evals ships
OR-Bench-Hard-1K. It does not: at the 0.18.0 pin its METADATA lists the
benchmark as an EXTERNAL eval living in a third-party repository. The
options were to install that package, vendor a task of the same shape, or
drop the comparison. This is the vendored one — no third-party install,
and, more importantly, it reads the SAME pinned CSV our judged control
reads (``bench-llm/or-bench`` ``or-bench-hard-1k.csv`` at the pins.yaml
revision), so the item overlap with our ``orb####`` rows is total rather
than approximate.

The SHAPE is XSTest's, and deliberately so: OR-Bench-Hard-1K asks the
same question XSTest asks (does the model refuse a benign-but-alarming
prompt?), and Inspect's XSTest answers it with a
``model_graded_qa(C/P/I)`` grader plus a ``refusal_rate`` metric. Both the
grader template and the metric are IMPORTED from inspect_evals' xstest
rather than re-typed, so the two comparison tasks are graded by one
rubric and a drift in it cannot land on only one of them.

The grader is the unsteered base route on the same proxy (P4) — the stock
default is ``openai/gpt-4o``, which this spec's no-external-API
constraint forbids. Numbers from it are labelled base-model-judged; they
are NOT comparable with our published sonnet-judged OR-Bench numbers, and
the report says so rather than differencing them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate, system_message

from inspect_evals.utils import create_stable_id
from inspect_evals.xstest.xstest import (refusal_rate, scorer_instructions,
                                         scorer_template)

#: Inspect's XSTest uses this exact system message; keeping it identical
#: is what makes the two refusal readouts comparable to each other.
SYSTEM_MESSAGE = "You are a helpful assistant."


def _sample(row: dict) -> Sample:
    return Sample(
        input=str(row["prompt"]),
        id=create_stable_id(row["prompt"], prefix="or_bench_hard_1k"),
        metadata={"paired_id": str(row["item_id"]),
                  "category": str(row.get("category", ""))},
    )


@task
def or_bench_hard_1k(scorer_model: Optional[str | Model] = None,
                     limit: int = 0,
                     pins: Optional[Any] = None,
                     positions: Optional[Sequence[int]] = None) -> Task:
    """OR-Bench-Hard-1K over the pinned CSV, graded XSTest-style.

    ``positions`` is an EXPLICIT list of 0-based indices into the pinned
    corpus order — the battery's frozen, seeded 200-item subset
    (``contracts.or_bench_subset_positions`` derives it and the caller passes
    it in, so this file holds no policy and the same list is what the run's
    identity hashed). ``None`` = the whole corpus. Out-of-range or duplicate
    positions are refused rather than clipped: a subset that silently shrank
    would still be reported as the frozen set.

    ``limit`` then heads whatever order results (the same order our
    ``orb####`` ids are minted in), so a smoke run's items are a prefix
    of the full run's rather than a different sample."""
    # lazy, like the sibling task: a config file should not pull the eval
    # stack in just to be read
    from sisteer.capeval.corpora import load_pins
    from sisteer.capeval.tiers.judged import or_bench_rows

    rows = or_bench_rows(pins if pins is not None else load_pins())
    if positions is not None:
        want = [int(i) for i in positions]
        if len(set(want)) != len(want):
            raise ValueError("or_bench_hard_1k: positions has duplicates — "
                             "refusing to run one prompt twice under a frozen subset")
        bad = [i for i in want if not 0 <= i < len(rows)]
        if bad:
            raise ValueError(
                f"or_bench_hard_1k: positions {bad[:5]} are outside 0..{len(rows) - 1} — "
                "the pinned corpus is not the one the subset was frozen against")
        rows = [rows[i] for i in sorted(want)]
    if limit:
        rows = rows[:int(limit)]
    return Task(
        dataset=MemoryDataset([_sample(r) for r in rows],
                              name="or_bench_hard_1k"),
        solver=[system_message(SYSTEM_MESSAGE), generate()],
        scorer=model_graded_qa(model=scorer_model, template=scorer_template,
                               instructions=scorer_instructions),
        metrics=[refusal_rate()],
        config=GenerateConfig(temperature=0, max_tokens=256),
    )
