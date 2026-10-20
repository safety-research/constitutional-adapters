"""Post-run StrongREJECT grading (JB5) — the authoritative ASR grader.

``BatchStrongREJECTGrader`` is a grading-host-side POST-STEP over the target responses
persisted by an attack. It is deliberately NOT a PyRIT ``Scorer`` and is NEVER
passed into a PyRIT attack or an in-loop scorer:

- the Batch API is async / grading-host-only (the batch org key lives only there), so
  it cannot gate a tree search / escalation live;
- keeping it strictly separate from the local in-loop ``LocalAttackScorers``
  (which ARE PyRIT scorers) is the two-scorer split the design rests on — the
  gap between the two is a REPORTED metric, not a bug.

It wraps ``scripts/judge_batch.grade_strongreject`` verbatim, inheriting its
resubmit → fixed-Sonnet-fallback → loud-completeness pipeline: a grader FAILURE
(``__blocked__`` / missing / unparseable) stays ``UNSCORED`` and is never scored
0 (0 is reserved for a validly-parsed ``refused==1`` rubric), and aggregation
FAILS LOUDLY if any expected response is still unscored (JB1)."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ holds judge_batch.py (not importable as a package); add it to path
# the same way scripts/jailbreak_run.py and the tests do.
_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _ensure_scripts_on_path() -> None:
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))


class BatchStrongREJECTGrader:
    """The authoritative StrongREJECT ASR grader — post-run ONLY.

    Not a PyRIT ``Scorer`` (no ``score_async`` / ``_score_piece_async``): a
    guard test asserts it is never an instance of ``pyrit.score.Scorer`` and so
    can never be handed to a PyRIT attack (JB5)."""

    def __init__(self, *, judge, sonnet_judge=None, poll_interval: int = 120) -> None:
        self.judge = judge
        self.sonnet_judge = sonnet_judge
        self.poll_interval = poll_interval

    def grade(self, items):
        """Grade ``items`` = list of ``(dataset, revision, goal, response)``.

        Returns ``{(dataset, revision, goal, response): score in [0, 1]}``.
        Raises loudly if any item stays UNSCORED after resubmit + Sonnet
        fallback (never a silent 0 — JB1)."""
        _ensure_scripts_on_path()
        from judge_batch import grade_strongreject

        return grade_strongreject(
            list(items), judge=self.judge, sonnet_judge=self.sonnet_judge,
            poll_interval=self.poll_interval,
        )
