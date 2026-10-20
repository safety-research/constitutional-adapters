"""A SKIPPED family is genuinely re-attempted on resume.

A skip and a partial completion are DIFFERENT states. A sub-env family that
skips (sub-env absent) must write NOTHING — no progress.jsonl, no
responses.jsonl — so a later resume with the family available regenerates every
goal, rather than a stray sentinel making resume record it as done. A genuinely
COMPLETED family, by contrast, is recorded and correctly skipped on resume.
"""

from __future__ import annotations

from dataclasses import dataclass

from sisteer.eval.attacks.amplegcg import run_amplegcg
from sisteer.eval.jailbreak import build_defense


@dataclass(frozen=True)
class _Goal:
    goal_id: str
    text: str
    category: str = "harmful"


def _goals(n):
    return [_Goal(f"g{i}", f"topic {i}") for i in range(n)]


def _defense():
    return build_defense(None, None, mode="none", positions="all",
                         batch_size=8, generate_fn=lambda r, **k: [f"resp:{x[-8:]}" for x in r])


def _gen(goal, seed):
    return [("a0", f"{goal.text} [REPLACED]")], {"n_prompts_generated": 1,
                                                 "n_generator_forwards": 1}


def test_skip_writes_no_state(tmp_path):
    out = tmp_path / "amplegcg_plus"
    res = run_amplegcg(_defense(), _goals(3), out_dir=out,
                      _available=lambda: (False, "sub-env absent"))
    assert res.get("skipped") is True
    # the skip left NO resume state — neither the progress sentinel nor responses
    assert not (out / "progress.jsonl").exists()
    assert not (out / "responses.jsonl").exists()


def test_skipped_family_is_reattempted_then_completion_is_recorded(tmp_path):
    out = tmp_path / "amplegcg_plus"
    d = _defense()
    # 1) skip: nothing generated, nothing recorded
    r1 = run_amplegcg(d, _goals(3), out_dir=out,
                     _available=lambda: (False, "sub-env absent"))
    assert r1.get("skipped") is True
    # 2) resume with the family AVAILABLE -> ALL goals re-attempted (not "done")
    r2 = run_amplegcg(d, _goals(3), out_dir=out,
                     _available=lambda: (True, "py"), _generate=_gen)
    assert r2["n_goals_run"] == 3
    # 3) a SECOND resume now finds them recorded complete -> skipped (0 redone).
    #    This is the state a skip must NOT be confused with.
    r3 = run_amplegcg(d, _goals(3), out_dir=out,
                     _available=lambda: (True, "py"), _generate=_gen)
    assert r3["n_goals_run"] == 0

