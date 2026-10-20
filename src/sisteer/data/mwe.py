"""Anthropic model-written evals (Perez et al.) — A/B behavior items.

The advanced-ai-risk sets are the extraction data of CAA and BiPO, so we
use them for the replication suite. Each JSONL row: question (with the
lettered choices embedded), answer_matching_behavior (" (A)"),
answer_not_matching_behavior. Choice texts are parsed out so BiPO-style
free-form preference pairs can be built from the same items.

Two verified gotchas in the human_generated_evals files (2026-08-11):

1. LABEL POLE: `answer_matching_behavior` marks the SAFE / behavior-AVERSE
   option, at least for power-seeking-inclination ("No, I shouldn't have
   control", 8/8 inspected) and survival-instinct ("Yes, I'm okay with
   being turned off"). Steering "+" along matching-minus-not_matching
   therefore pushes toward the SAFE pole. Method-level claims
   (monotonicity, range, reliability) are polarity-agnostic, but any
   semantic reading of propensities must use this pole. A trait-inducing
   system prompt LOWERS p(matching) here — that is the prompt working as
   written, not an inversion.
2. MULTI-CHOICE CONTAMINATION: 26-40% of rows have a (C) (sometimes (D))
   option; naive A/B parsing corrupts them (the (C) text leaks into
   choice_b, and non-A/B matching labels are meaningless in a two-way
   softmax). load_mwe drops these rows by default (strict_ab=True).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass


@dataclass
class ABItem:
    question: str
    matching: str  # " (A)" / " (B)" (normalized to leading space form)
    not_matching: str
    choice_a: str | None
    choice_b: str | None

    @property
    def matching_letter(self) -> str:
        return "A" if "A" in self.matching else "B"

    @property
    def matching_text(self) -> str | None:
        return self.choice_a if self.matching_letter == "A" else self.choice_b

    @property
    def not_matching_text(self) -> str | None:
        return self.choice_b if self.matching_letter == "A" else self.choice_a


_CHOICES_RE = re.compile(
    r"\(A\)\s*(?P<a>.*?)\s*\(B\)\s*(?P<b>.*?)\s*$", re.DOTALL
)


def parse_mwe_lines(lines: list[str], strict_ab: bool = True) -> list[ABItem]:
    items = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        q = row["question"]
        if strict_ab:
            # Drop rows with more than two options or non-A/B labels: the
            # (C) text corrupts choice_b and a two-way softmax over {A, B}
            # is meaningless for a (C)-labeled answer.
            labels = row["answer_matching_behavior"] + row[
                "answer_not_matching_behavior"
            ]
            if "(C)" in q or "(D)" in q or "C" in labels or "D" in labels:
                continue
        m = _CHOICES_RE.search(q)
        choice_a = m.group("a").strip() if m else None
        choice_b = m.group("b").strip() if m else None

        def norm(ans: str) -> str:
            return " (A)" if "A" in ans else " (B)"

        items.append(
            ABItem(
                question=q,
                matching=norm(row["answer_matching_behavior"]),
                not_matching=norm(row["answer_not_matching_behavior"]),
                choice_a=choice_a,
                choice_b=choice_b,
            )
        )
    return items


def load_mwe(
    behavior: str, source: str = "human_generated_evals", strict_ab: bool = True
) -> list[ABItem]:
    """behavior e.g. 'power-seeking-inclination', 'survival-instinct'."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Anthropic/model-written-evals",
        repo_type="dataset",
        filename=f"advanced-ai-risk/{source}/{behavior}.jsonl",
    )
    with open(path) as f:
        return parse_mwe_lines(f.readlines(), strict_ab=strict_ab)


def balance_letters(items: list[ABItem], seed: int = 0) -> list[ABItem]:
    """Subsample so matching=A and matching=B are 50/50.

    The raw MWE advanced-ai-risk files are letter-imbalanced (power-seeking
    38% A, survival-instinct 62% A). Unbalanced, the mean-diff vector keeps
    an uncancelled answer-letter component and the A/B propensity metric is
    letter-biased (Tan et al.'s spurious position/token factors); CAA's own
    repo datasets are rebalanced."""
    rng = random.Random(seed)
    a = [it for it in items if it.matching_letter == "A"]
    b = [it for it in items if it.matching_letter == "B"]
    n = min(len(a), len(b))
    rng.shuffle(a)
    rng.shuffle(b)
    out = a[:n] + b[:n]
    rng.shuffle(out)
    return out


def split_items(
    items: list[ABItem], n_train: int, n_test: int, seed: int = 0
) -> tuple[list[ABItem], list[ABItem]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train : n_train + n_test]
