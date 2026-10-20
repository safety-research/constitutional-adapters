"""Flag incoherent / repetitious completions (user instruction, 2026-08-27).

WHY THIS IS NOT A FILTER. Small models fall into repetition loops under greedy
decode, and a completeness-scoring judge (StrongREJECT) reads a loop as a
successful defense -- so a degenerate arm looks SAFER than it is. Measured on
the Llama-3.2-1B two-SI pool (89,372 rows, identical prompts to every other
model in the set):

    safety_refusal  0.77%      none            1.50%
    si_constitution 1.77%      si_irrelevant   3.21%      (4.2x spread)

with `grok420/none` at 38.2% and `codeattack/si_irrelevant` at 20.4%. The same
prompts through gemma-4-E2B: 0.02-0.04%, spread 0.02% -- so it is the 1B
model, not the generation path.

Dropping these rows would fix one model's numbers and break the thing the
whole set exists for: the sibling pools were built without such a filter, so a
filter here makes cross-model comparison invalid. We therefore FLAG and let
the consumer decide, per the user's call. Raising the token cap is not an
alternative -- this is non-termination, not a long answer: 4096 -> 9216 left
77.9% of the remaining truncations still looping.

Both detectors below were validated against hand-inspected examples on that
pool; the tail-block rule catches the `-- | -- | --` shape, the line rule
catches repeated sentences/bullets that the tail rule misses.
"""
from __future__ import annotations

import collections
from typing import Dict, Optional

# Last 60 chars repeated >=3x within the final 2,000. Short window on purpose:
# a loop that started and then stopped is not what we are flagging.
TAIL_WINDOW = 2000
TAIL_BLOCK = 60
TAIL_MIN_REPEATS = 3
TAIL_MIN_LEN = 200

# One line accounting for >=25% of all non-empty lines. Needs enough lines to
# mean anything -- a 3-line answer with a repeated line is not degenerate.
LINE_MIN_LINES = 8
LINE_MAX_SHARE = 0.25


def _tail_loop(text: str) -> bool:
    tail = text[-TAIL_WINDOW:]
    if len(tail) < TAIL_MIN_LEN:
        return False
    return tail.count(tail[-TAIL_BLOCK:]) >= TAIL_MIN_REPEATS


def _line_loop(text: str) -> bool:
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if len(lines) < LINE_MIN_LINES:
        return False
    top = collections.Counter(lines).most_common(1)[0][1]
    return top / len(lines) >= LINE_MAX_SHARE


def classify(text: Optional[str]) -> Dict[str, object]:
    """Return ``{"degenerate": bool, "degenerate_reason": str|None}``.

    ``reason`` is ``"tail_loop"``, ``"line_loop"``, ``"both"`` or ``None``, so
    a reader can tell WHICH shape fired rather than trusting one bit.
    """
    t = text or ""
    tail, line = _tail_loop(t), _line_loop(t)
    if tail and line:
        reason = "both"
    elif tail:
        reason = "tail_loop"
    elif line:
        reason = "line_loop"
    else:
        reason = None
    return {"degenerate": bool(tail or line), "degenerate_reason": reason}
