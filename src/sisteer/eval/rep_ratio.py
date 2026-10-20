"""The campaign's ONE ``rep_ratio`` — the greedy full22 join's looping proxy, importable.

WHY THIS FILE. The definition was born in ``~/modal-stage/sdfbase/fullscale/full22_join_scores.py``, a SCRIPT (a run loop
at module level, imports of a worktree that no longer exists) that cannot be imported, and was then re-typed ONCE, in
``scripts/alphasteer_f22/goal_level_verdicts.py``, from where ``sisteer.eval.static_grading.degeneration_block`` and
``sisteer.eval.degeneracy_union`` imported it. Two typed bodies of one definition can drift. This module is the single
importable home: every live caller — the shared grading body, the union, the alphasteer_f22 lane script — imports it from
here (collapsed 2026-09-13). The origin script stays FROZEN as a historical record: its sha is pinned in the published
flags manifest (``run-state/full22/degeneracy/flags_manifest.json`` ``definition_source_sha256``) and in
``degeneracy_union.DEFINITION_SOURCES["rep_ratio_loop"]["definition_origin"]``, and
``degeneracy_union.verify_definition_sources`` compares THIS body with the origin's by AST on every run.

The body below is the ``goal_level_verdicts`` copy moved verbatim — the body the union was validated with (97/97 exact
against four origin lanes, 2026-09-13). It differs from the origin only by ``str(text).split()`` vs ``t.split()``,
identical on the string input every caller guarantees (each refuses a non-string response before calling).

Definition: 1 - (distinct whitespace-token 4-grams / all 4-grams); 0.0 under 8 words. Looping := ``rep_ratio > 0.5``.
"""
from __future__ import annotations

LOOP_REP_RATIO = 0.5


def rep_ratio(text: str) -> float:
    """The wave join's looping proxy: 1 - distinct-4-gram ratio over whitespace tokens (0 under 8 words)."""
    w = str(text).split()
    if len(w) < 8:
        return 0.0
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g)


__all__ = ["LOOP_REP_RATIO", "rep_ratio"]
